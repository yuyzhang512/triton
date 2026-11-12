#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "third_party/amd/include/TritonAMDGPUToLLVM/PatternTritonAMDGPUToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// One request covers 8 VGPRs: one gfx1250 WMMA fragment or accumulator tile.
constexpr unsigned kRegsPerPin = 8;

std::string typeSuffix(Type elemTy) {
  if (elemTy.isF32())
    return "f32";
  if (elemTy.isF16())
    return "f16";
  if (elemTy.isBF16())
    return "bf16";
  return "i" + std::to_string(elemTy.getIntOrFloatBitWidth());
}

// Pass each thread's elements of `src`, 8 VGPRs at a time, through
// `name`.v<N><type>(chunk, args(first VGPR of the chunk)...) and return the
// tensor rebuilt from the results.
template <typename ArgsFn>
FailureOr<Value> passThroughChunks(Operation *op, Value src,
                                   RankedTensorType tensorTy, StringRef name,
                                   const LLVMTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter,
                                   ArgsFn args) {
  Location loc = op->getLoc();
  TritonLLVMOpBuilder b(loc, rewriter);
  Type elemTy = typeConverter->convertType(tensorTy.getElementType());
  if (!elemTy.isIntOrFloat())
    return failure();
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  if (bits != 16 && bits != 32)
    return failure();
  unsigned perReg = 32 / bits;
  SmallVector<Value> vals = unpackLLElements(loc, src, rewriter);
  if (vals.size() % perReg)
    return failure();

  unsigned chunk = kRegsPerPin * perReg;
  SmallVector<Value> out;
  out.reserve(vals.size());
  for (size_t i = 0; i < vals.size(); i += chunk) {
    unsigned n = std::min<size_t>(chunk, vals.size() - i);
    Type vecTy = vec_ty(elemTy, n);
    Value v = b.undef(vecTy);
    for (unsigned j = 0; j < n; ++j)
      v = b.insert_element(vecTy, v, vals[i + j], b.i32_val(j));
    SmallVector<Value> operands{v};
    SmallVector<Type> operandTys{vecTy};
    for (Value a : args(b, i / perReg)) {
      operands.push_back(a);
      operandTys.push_back(a.getType());
    }
    std::string fnName =
        (name + ".v" + Twine(n) + typeSuffix(elemTy)).str();
    auto fnTy = LLVM::LLVMFunctionType::get(vecTy, operandTys);
    LLVM::LLVMFuncOp fn =
        triton::gpu::appendOrGetExternFuncOp(rewriter, op, fnName, fnTy);
    Value res = LLVM::CallOp::create(rewriter, loc, fn, operands).getResult();
    for (unsigned j = 0; j < n; ++j)
      out.push_back(b.extract_element(elemTy, res, b.i32_val(j)));
  }
  return packLLElements(loc, typeConverter, out, rewriter, tensorTy);
}

struct PinVGPROpConversion : public ConvertOpToLLVMPattern<amdgpu::PinVGPROp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::PinVGPROp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int32_t> bases = op.getBases();
    if (bases.empty())
      return rewriter.notifyMatchFailure(op, "no base register");
    auto res = passThroughChunks(
        op, adaptor.getSrc(), cast<RankedTensorType>(op.getType()),
        "llvm.amdgcn.pin.vgpr", getTypeConverter(), rewriter,
        [&](TritonLLVMOpBuilder &b, int64_t reg) {
          int64_t chunk = reg / kRegsPerPin;
          int64_t first = bases.size() == 1 ? bases[0] + reg
                          : chunk < (int64_t)bases.size()
                              ? bases[chunk]
                              : bases.back() + (chunk - bases.size() + 1) *
                                                   kRegsPerPin;
          return SmallVector<Value>{b.i32_val(first)};
        });
    if (failed(res))
      return rewriter.notifyMatchFailure(
          op, "elements must be 16 or 32 bits and fill whole VGPRs");
    rewriter.replaceOp(op, *res);
    return success();
  }
};

// The loads producing the tensor are lowered by now, so the request travels
// with the data; the backend finds the loads from it.
struct PinLDSAddressOpConversion
    : public ConvertOpToLLVMPattern<amdgpu::PinLDSAddressOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::PinLDSAddressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t first = op.getFirst(), count = op.getCount();
    auto res = passThroughChunks(
        op, adaptor.getSrc(), cast<RankedTensorType>(op.getType()),
        "llvm.amdgcn.pin.lds.addr", getTypeConverter(), rewriter,
        [&](TritonLLVMOpBuilder &b, int64_t) {
          return SmallVector<Value>{b.i32_val(first), b.i32_val(count)};
        });
    if (failed(res))
      return rewriter.notifyMatchFailure(
          op, "elements must be 16 or 32 bits and fill whole VGPRs");
    rewriter.replaceOp(op, *res);
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populatePinVGPROpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit) {
  patterns.add<PinVGPROpConversion, PinLDSAddressOpConversion>(typeConverter,
                                                              benefit);
}

} // namespace mlir::triton::AMD
