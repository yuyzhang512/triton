// RUN: triton-opt %s -split-input-file --tritonllvm-vectorize | FileCheck %s

// A causal mask compares a row against T + c, one constant c per register
// (here c = 0 and 8, joined to the lane part with `or disjoint`). The rewrite
// compares row - T against the constant instead: one subtraction for all
// registers and no per-register index for LICM to hoist.

// CHECK-LABEL: llvm.func @causal_mask
// CHECK-DAG: %[[C8:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK: %[[T:.*]] = llvm.add %{{.*}}, %{{.*}} : i32
// CHECK: %[[D:.*]] = llvm.sub %arg0, %{{.*}} : i32
// CHECK-NOT: llvm.sub
// CHECK: llvm.icmp "sge" %[[D]], %[[C8]] : i32
llvm.func @causal_mask(%row: i32, %start: i32, %base: i32, %delta: i32) -> i1 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c8 = llvm.mlir.constant(8 : i32) : i32
  %o0 = llvm.or disjoint %base, %c0 : i32
  %o1 = llvm.or disjoint %base, %c8 : i32
  %k0 = llvm.add %start, %o0 : i32
  %k1 = llvm.add %start, %o1 : i32
  %b0 = llvm.add %k0, %delta : i32
  %b1 = llvm.add %k1, %delta : i32
  %m0 = llvm.icmp "sge" %row, %b0 : i32
  %m1 = llvm.icmp "sge" %row, %b1 : i32
  %m = llvm.and %m0, %m1 : i1
  llvm.return %m : i1
}

// -----

// Unsigned compares are left alone: row - T can be negative.
// A plain or (not disjoint) is not split.

// CHECK-LABEL: llvm.func @left_alone
// CHECK-NOT: llvm.sub
// CHECK: llvm.icmp "ult" %arg0, %{{.*}} : i32
// CHECK: llvm.icmp "sge" %arg0, %{{.*}} : i32
llvm.func @left_alone(%row: i32, %start: i32, %base: i32) -> i1 {
  %c8 = llvm.mlir.constant(8 : i32) : i32
  %o = llvm.or disjoint %base, %c8 : i32
  %k = llvm.add %start, %o : i32
  %u = llvm.icmp "ult" %row, %k : i32
  %x = llvm.or %base, %c8 : i32
  %kx = llvm.add %start, %x : i32
  %s = llvm.icmp "sge" %row, %kx : i32
  %m = llvm.and %u, %s : i1
  llvm.return %m : i1
}
