// RUN: triton-opt %s -split-input-file --tritonamdgpu-plumb-register-plan="mode=homes report=true" -verify-diagnostics | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonamdgpu-plumb-register-plan="mode=report" | FileCheck %s --check-prefix=REPORT

// A dot operand carried around the loop and reloaded in each iteration gets one
// AGPR home: its initial value, the block argument, the yielded value and the
// loop result are pinned to the same AGPRs.

#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @carried
  // CHECK: %[[I:.*]] = amdg.pin_vgpr %arg1, [0] {agpr}
  // CHECK: scf.for {{.*}} iter_args(%[[B:.*]] = %[[I]],
  // CHECK: amdg.pin_vgpr %[[B]], [0] {agpr}
  // CHECK: %[[N:.*]] = arith.addf
  // CHECK: amdg.pin_vgpr %[[N]], [0] {agpr}
  // CHECK: amdg.pin_vgpr %{{.*}}#0, [0] {agpr}
  // REPORT-LABEL: tt.func @carried
  // REPORT-NOT: amdg.pin_vgpr
  tt.func @carried(%a: tensor<16x32xf16, #dotA>, %b0: tensor<32x16xf16, #dotB>,
                   %b1: tensor<32x16xf16, #dotB>, %n: i32) -> (tensor<32x16xf16, #dotB>, tensor<16x16xf32, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    // expected-remark @+1 {{plumb-register-plan: peak}}
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%b = %b0, %acc = %zero)
        -> (tensor<32x16xf16, #dotB>, tensor<16x16xf32, #mma>) : i32 {
      %d = tt.dot %a, %b, %acc : tensor<16x32xf16, #dotA> * tensor<32x16xf16, #dotB> -> tensor<16x16xf32, #mma>
      %next = arith.addf %b1, %b1 : tensor<32x16xf16, #dotB>
      scf.yield %next, %d : tensor<32x16xf16, #dotB>, tensor<16x16xf32, #mma>
    }
    tt.return %r#0, %r#1 : tensor<32x16xf16, #dotB>, tensor<16x16xf32, #mma>
  }
}
