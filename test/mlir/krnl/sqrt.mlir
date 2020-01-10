// RUN: onnf-opt --shape-inference --lower-all-llvm %s -split-input-file | FileCheck %s

func @test_sqrt(%arg0 : f32) -> f32 {
  %0 = "krnl.sqrt"(%arg0) : (f32) -> f32
  "std.return"(%0) : (f32) -> ()

  // CHECK: llvm.func @llvm.sqrt.f32(!llvm.float) -> !llvm.float
  // CHECK: llvm.func @test_sqrt(%arg0: !llvm.float) -> !llvm.float {
  // CHECK: [[RES:%.+]] = llvm.call @llvm.sqrt.f32(%arg0) : (!llvm.float) -> !llvm.float
  // CHECK: llvm.return [[RES]] : !llvm.float
}
