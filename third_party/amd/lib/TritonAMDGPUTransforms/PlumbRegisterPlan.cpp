// [PlumbCompiler] Register planning on TTGIR: one home per loop-carried MFMA
// operand.
//
// On gfx950 a wave has 256 VGPRs and 256 AGPRs. MFMA reads its A/B operands
// from either file, and the loads that produce them can write AGPRs directly.
// A tensor carried around the main loop is four SSA values: the value that
// initializes it, the loop's block argument, the value yielded for the next
// iteration and the loop result read after the loop. LLVM gives them separate
// virtual registers; what it cannot coalesce becomes copies at the loop exit,
// before the peeled drain. This pass makes the four one object and pins each to
// the same AGPRs (amdg.pin_vgpr {agpr}).
//
// Measured on gfx950 MoE GEMMs, homes pay only when the kernel runs out of
// registers, so `auto` places them only above a register-pressure estimate:
// the peak number of tensor registers per thread live in the loop body (block
// arguments, loop-invariant tensors used inside, and values defined in the
// body until their last use, nested uses counted at their top-level op).
//
// Options: mode = report | homes | auto, pressure (auto threshold), max-regs
// (largest object to place), agpr-base (first AGPR), report (emit remarks).

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace ttg = mlir::triton::gpu;
namespace amdgpu = mlir::triton::amdgpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPLUMBREGISTERPLAN
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// 32-bit registers per thread a tensor value occupies, from Triton's layout
// code (a dot operand can be replicated across warps).
int64_t registersOf(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || !ty.getEncoding())
    return 0;
  Type elem = ty.getElementType();
  if (!elem.isIntOrFloat())
    return 0;
  int64_t elems = ttg::getTotalElemsPerThread(ty);
  int64_t bits = elem.getIntOrFloatBitWidth();
  return (elems * bits + 31) / 32;
}

bool isDotOperand(Value v) {
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  return ty && isa_and_nonnull<ttg::DotOperandEncodingAttr>(ty.getEncoding());
}

int64_t countDots(Operation *op) {
  int64_t n = 0;
  op->walk([&](Operation *o) {
    StringRef name = o->getName().getStringRef();
    if (name.contains("dot") || name.contains("mfma"))
      ++n;
  });
  return n;
}

struct LoopFacts {
  scf::ForOp loop;
  SmallVector<Operation *> body;          // top-level ops, in order
  DenseMap<Operation *, int64_t> index;   // top-level op -> position
  int64_t peak = 0;                       // peak live tensor registers
  int64_t carried = 0;                    // registers of the block arguments
};

// Position of the top-level body op that contains `op`, or -1.
int64_t topIndex(LoopFacts &f, Operation *op) {
  Block *body = f.loop.getBody();
  Operation *top = body->findAncestorOpInBlock(*op);
  if (!top)
    return -1;
  auto it = f.index.find(top);
  return it == f.index.end() ? -1 : it->second;
}

LoopFacts analyze(scf::ForOp loop) {
  LoopFacts f;
  f.loop = loop;
  for (Operation &op : *loop.getBody()) {
    f.index[&op] = f.body.size();
    f.body.push_back(&op);
  }
  int64_t n = f.body.size();
  // Live interval [start, end] of every tensor value used in the body.
  DenseMap<Value, std::pair<int64_t, int64_t>> live;
  auto touch = [&](Value v, int64_t at) {
    if (registersOf(v) == 0)
      return;
    int64_t start = -1;
    if (Operation *def = v.getDefiningOp()) {
      int64_t d = topIndex(f, def);
      start = d >= 0 ? d : -1; // defined outside: live through the loop
      if (d < 0)
        at = n;
    }
    auto it = live.try_emplace(v, start, at).first;
    it->second.second = std::max(it->second.second, at);
  };
  loop.getBody()->walk([&](Operation *op) {
    int64_t at = topIndex(f, op);
    for (Value v : op->getOperands())
      touch(v, at);
  });
  for (BlockArgument arg : loop.getRegionIterArgs()) {
    f.carried += registersOf(arg);
    if (!live.count(arg) && registersOf(arg))
      live.try_emplace(arg, -1, -1);
  }
  // Values defined in the body but never used still occupy registers at
  // their definition.
  for (Operation *op : f.body)
    for (Value r : op->getResults())
      if (registersOf(r) && !live.count(r))
        live.try_emplace(r, f.index[op], f.index[op]);
  std::vector<int64_t> load(n + 2, 0);
  for (auto &[v, iv] : live) {
    int64_t regs = registersOf(v);
    for (int64_t i = std::max<int64_t>(iv.first, 0);
         i <= std::min<int64_t>(iv.second, n); ++i)
      load[i] += regs;
  }
  for (int64_t x : load)
    f.peak = std::max(f.peak, x);
  return f;
}

struct Home {
  unsigned arg;
  int64_t regs;
  int64_t base;
};

Value pin(OpBuilder &b, Location loc, Value v, int64_t base) {
  return amdgpu::PinVGPROp::create(b, loc, v.getType(), v,
                                   b.getDenseI32ArrayAttr({(int32_t)base}),
                                   b.getUnitAttr())
      .getResult();
}

} // namespace

struct TritonAMDGPUPlumbRegisterPlanPass
    : impl::TritonAMDGPUPlumbRegisterPlanBase<
          TritonAMDGPUPlumbRegisterPlanPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // The main loops: scf.for ops with dots in their own body, not loops
    // nested in another such loop's body only for bookkeeping.
    SmallVector<scf::ForOp> loops;
    mod.walk([&](scf::ForOp loop) {
      if (countDots(loop) > 0 &&
          !llvm::any_of(loop.getBody()->getOperations(), [](Operation &op) {
            return isa<scf::ForOp>(op) && countDots(&op) > 0;
          }))
        loops.push_back(loop);
    });
    int64_t next = agprBase;
    for (scf::ForOp loop : loops) {
      LoopFacts f = analyze(loop);
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      SmallVector<Home> homes;
      std::string why;
      llvm::raw_string_ostream os(why);
      bool place = mode == "homes" || (mode == "auto" && f.peak >= pressure);
      for (auto [k, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
        if (!isDotOperand(arg))
          continue;
        int64_t regs = registersOf(arg);
        Value y = yield.getOperand(k);
        os << " arg" << k << "=" << regs << "r";
        if (regs > maxRegs) {
          os << "(>max)";
          continue;
        }
        if (y == arg || !y.getDefiningOp() ||
            !loop->isAncestor(y.getDefiningOp())) {
          os << "(not reloaded)";
          continue;
        }
        // The yielded value must be defined after the argument's last read.
        int64_t last = -1;
        for (Operation *user : arg.getUsers())
          if (user != yield.getOperation())
            last = std::max(last, topIndex(f, user));
        if (topIndex(f, y.getDefiningOp()) < last) {
          os << "(overlaps)";
          continue;
        }
        int64_t align = regs >= 4 ? 4 : regs;
        next = (next + align - 1) / align * align;
        if (next + regs > 256) {
          os << "(no AGPRs left)";
          continue;
        }
        homes.push_back({(unsigned)k, regs, next});
        next += regs;
      }
      if (report)
        loop.emitRemark() << "plumb-register-plan: peak " << f.peak
                          << " regs, carried " << f.carried
                          << ", dot operands:" << why << "; "
                          << (place ? "placing " : "not placing ")
                          << homes.size() << " AGPR homes (mode " << mode
                          << ", pressure " << static_cast<int64_t>(pressure) << ")";
      if (!place)
        continue;
      Location loc = loop.getLoc();
      for (Home &h : homes) {
        BlockArgument arg = loop.getRegionIterArgs()[h.arg];
        // Initial value, right before the loop.
        OpBuilder b(loop);
        OpOperand &init = loop.getInitArgsMutable()[h.arg];
        init.set(pin(b, loc, init.get(), h.base));
        // Block argument, at the top of the body.
        b.setInsertionPointToStart(loop.getBody());
        Value pa = pin(b, loc, arg, h.base);
        arg.replaceAllUsesExcept(pa, pa.getDefiningOp());
        // Yielded value, right after its definition.
        Value y = yield.getOperand(h.arg);
        b.setInsertionPointAfter(y.getDefiningOp());
        Value py = pin(b, loc, y, h.base);
        y.replaceAllUsesExcept(py, py.getDefiningOp());
        // Loop result, right after the loop.
        Value r = loop.getResult(h.arg);
        b.setInsertionPointAfter(loop);
        Value pr = pin(b, loc, r, h.base);
        r.replaceAllUsesExcept(pr, pr.getDefiningOp());
      }
    }
  }
};

} // namespace mlir
