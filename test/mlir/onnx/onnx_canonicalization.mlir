// RUN: onnf-opt --canonicalize %s -split-input-file | FileCheck %s

func @test_matmul_add_simplification(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-LABEL: test_matmul_add_simplification
  // CHECK: %{{[0-9]+}} = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%1) : (tensor<10x10xf32>) -> ()
}

// onnx.MatMul ops with more than one result uses should not get fused
// CHECK-LABEL: func @test_sigmoid_add(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func @test_sigmoid_add(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>, %a2: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK: %{{[0-9]+}} = "onnx.MatMul"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.MatMul"(%a0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %a2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %a1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %3 = "onnx.Add"(%1, %2) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%3) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL: @test_identity_identity(%{{.*}}: tensor<10x10xf32>, %{{.*}}: tensor<10x10xf32>) -> tensor<10x10xf32>
func @test_identity_identity(%a0: tensor<10x10xf32>, %a1: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %0 = "onnx.Identity"(%a0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Identity"(%a1) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "std.return"(%2) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL: @test_reducel1(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel1(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL1"(%arg0) {axes=[1], keepdims = 0 : i32} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[ABS:%.+]] =  "onnx.Abs"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[ABS]]) {axes = [1], keepdims = 0 : i32} : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducel2(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducel2(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL2"(%arg0) {axes=[1], keepdims = 0 : i32} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[MUL:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[MUL]]) {axes = [1], keepdims = 0 : i32} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQRT:%.+]] =  "onnx.Sqrt"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducelogsum(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsum(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSum"(%arg0) {axes=[1], keepdims = 0 : i32} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"(%arg0) {axes = [1], keepdims = 0 : i32} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducelogsumexp(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducelogsumexp(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0) {axes=[1], keepdims = 0 : i32} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[EXP:%.+]] =  "onnx.Exp"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]]) {axes = [1], keepdims = 0 : i32} : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// CHECK-LABEL: @test_reducesumsquare(%{{.*}}: tensor<?x?x?xf32>) -> tensor<*xf32>
func @test_reducesumsquare(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumSquare"(%arg0) {axes=[1], keepdims = 0 : i32} : (tensor<?x?x?xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[SQUARE:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[SQUARE]]) {axes = [1], keepdims = 0 : i32} : (tensor<*xf32>) -> tensor<*xf32>
}
