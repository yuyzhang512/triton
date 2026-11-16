// RUN: triton-opt %s -split-input-file --tritonllvm-vectorize | FileCheck %s

// A loop-carried accumulator read only as aligned 4-wide slices that feed MFMA
// accumulators (one per tile), and rebuilt by concatenating the tiles, is
// carried as 4-wide pieces.

// CHECK-LABEL: llvm.func @sliced_acc
// CHECK: ^bb1(%{{.*}}: i32, %{{.*}}: vector<4xf32>, %{{.*}}: vector<4xf32>):
// CHECK-NOT: vector<8xf32>
// CHECK: llvm.return
llvm.func @sliced_acc(%n: i32, %a: vector<8xf16>, %b: vector<8xf16>) -> f32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %zero = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
  llvm.br ^bb1(%c0, %zero : i32, vector<8xf32>)
^bb1(%i: i32, %acc: vector<8xf32>):
  %lo = llvm.shufflevector %acc, %acc [0, 1, 2, 3] : vector<8xf32>
  %hi = llvm.shufflevector %acc, %acc [4, 5, 6, 7] : vector<8xf32>
  %lo2 = rocdl.mfma.f32.16x16x32.f16 %a, %b, %lo, 0, 0, none : (vector<8xf16>, vector<8xf16>, vector<4xf32>) -> vector<4xf32>
  %hi2 = rocdl.mfma.f32.16x16x32.f16 %a, %b, %hi, 0, 0, none : (vector<8xf16>, vector<8xf16>, vector<4xf32>) -> vector<4xf32>
  %next = llvm.shufflevector %lo2, %hi2 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>
  %i2 = llvm.add %i, %c1 : i32
  %done = llvm.icmp "sge" %i2, %n : i32
  llvm.cond_br %done, ^bb2, ^bb1(%i2, %next : i32, vector<8xf32>)
^bb2:
  %e = llvm.extractelement %lo2[%c0 : i32] : vector<4xf32>
  %f = llvm.extractelement %hi2[%c0 : i32] : vector<4xf32>
  %r = llvm.fadd %e, %f : f32
  llvm.return %r : f32
}

// -----

// An argument also used as a whole vector stays as it is.

// CHECK-LABEL: llvm.func @whole_use
// CHECK: ^bb1(%{{.*}}: i32, %{{.*}}: vector<8xf32>):
llvm.func @whole_use(%n: i32, %s: vector<8xf32>) -> vector<8xf32> {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %zero = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
  llvm.br ^bb1(%c0, %zero : i32, vector<8xf32>)
^bb1(%i: i32, %acc: vector<8xf32>):
  %scaled = llvm.fmul %acc, %s : vector<8xf32>
  %i2 = llvm.add %i, %c1 : i32
  %done = llvm.icmp "sge" %i2, %n : i32
  llvm.cond_br %done, ^bb2, ^bb1(%i2, %scaled : i32, vector<8xf32>)
^bb2:
  llvm.return %scaled : vector<8xf32>
}

// -----

// Slices that feed something other than an MFMA accumulator stay unsplit.

// CHECK-LABEL: llvm.func @sliced_other
// CHECK: ^bb1(%{{.*}}: i32, %{{.*}}: vector<8xf32>):
llvm.func @sliced_other(%n: i32, %x: vector<4xf32>) -> f32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %zero = llvm.mlir.constant(dense<0.000000e+00> : vector<8xf32>) : vector<8xf32>
  llvm.br ^bb1(%c0, %zero : i32, vector<8xf32>)
^bb1(%i: i32, %acc: vector<8xf32>):
  %lo = llvm.shufflevector %acc, %acc [0, 1, 2, 3] : vector<8xf32>
  %hi = llvm.shufflevector %acc, %acc [4, 5, 6, 7] : vector<8xf32>
  %lo2 = llvm.fadd %lo, %x : vector<4xf32>
  %hi2 = llvm.fadd %hi, %x : vector<4xf32>
  %next = llvm.shufflevector %lo2, %hi2 [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xf32>
  %i2 = llvm.add %i, %c1 : i32
  %done = llvm.icmp "sge" %i2, %n : i32
  llvm.cond_br %done, ^bb2, ^bb1(%i2, %next : i32, vector<8xf32>)
^bb2:
  %e = llvm.extractelement %lo2[%c0 : i32] : vector<4xf32>
  %f = llvm.extractelement %hi2[%c0 : i32] : vector<4xf32>
  %r = llvm.fadd %e, %f : f32
  llvm.return %r : f32
}
