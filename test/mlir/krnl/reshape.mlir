// RUN: onnf-opt --shape-inference --lower-frontend --lower-krnl --lower-all-llvm %s -split-input-file | FileCheck %s

func @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi32>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK: llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm<"i8*">, !llvm<"i8*">, !llvm.i64, !llvm.i1)
  // CHECK: [[RES:%.+]] = llvm.insertvalue {{.*}}[4, 3] : !llvm<"{ float*, float*, i64, [4 x i64], [4 x i64] }">
  // CHECK: [[EXT_VAL_0:%.+]] = llvm.extractvalue [[RES]][1] : !llvm<"{ float*, float*, i64, [4 x i64], [4 x i64] }">
  // CHECK: [[DST:%.+]] = llvm.bitcast [[EXT_VAL_0]] : !llvm<"float*"> to !llvm<"i8*">
  // CHECK: [[EXT_VAL_1:%.+]] = llvm.extractvalue %0[1] : !llvm<"{ float*, float*, i64, [2 x i64], [2 x i64] }">
  // CHECK: [[SRC:%.+]] = llvm.bitcast [[EXT_VAL_1]] : !llvm<"float*"> to !llvm<"i8*">
  // CHECK: [[SIZE:%.+]] = llvm.sext %{{.*}} : !llvm.i64 to !llvm.i64
  // CHECK: [[VOLATILE:%.+]] = llvm.mlir.constant(0 : i1) : !llvm.i1
  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[DST]], [[SRC]], [[SIZE]], [[VOLATILE]]) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i64, !llvm.i1) -> !llvm.void
  // CHECK: llvm.return [[RES]] : !llvm<"{ float*, float*, i64, [4 x i64], [4 x i64] }">
}
