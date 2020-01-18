//===- ONNXCombine.cpp - ONNX High Level Optimizer ------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ONNX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include <numeric>
#include "src/dialect/onnx/onnx_ops.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/onnx_combine.inc"

// Due to a limitation of ODS that "all ODS-generated `build()` methods require
// specifying the result type(s), unless the op has known traits like
// `SameOperandsAndResultType` that we can use to auto-generate a `build()`
// method with result type deduction", we will to write some patterns manually
// until we find a better way.
//
// More information about the limitation can be found here:
// https://github.com/llvm/llvm-project/blob/master/mlir/docs/DeclarativeRewrites.md#building-operations
//

//===----------------------------------------------------------------------===//
// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
//===----------------------------------------------------------------------===//
struct ReduceSumSquareOpPattern : public RewritePattern {
  ReduceSumSquareOpPattern(MLIRContext *context)
      : RewritePattern("onnx.ReduceSumSquare", {"onnx.Mul", "onnx.ReduceSum"},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range X(op->getOperands());

    // Match
    auto opODS = dyn_cast_or_null<ONNXReduceSumSquareOp>(op);
    auto opODS_X = opODS.getODSOperands(0);
    auto opODS_Y = opODS.getODSResults(0);
    auto opODS_Attrs = opODS.getAttrs();

    // Rewrite
    auto loc = op->getLoc();
    ONNXMulOp mulOp;
    {
      Value X = (*opODS_X.begin());
      auto elementType = X->getType().cast<TensorType>().getElementType();
      mulOp = rewriter.create<ONNXMulOp>(
          loc, UnrankedTensorType::get(elementType), X, X);
    }

    ONNXReduceSumOp sumOp;
    {
      SmallVector<Value, 4> values;
      SmallVector<NamedAttribute, 4> attrs;
      SmallVector<Type, 4> types;
      values.push_back((*mulOp.getODSResults(0).begin()));
      for (auto attr : opODS_Attrs) {
        attrs.push_back(attr);
      }
      for (auto v : opODS_Y) {
        types.push_back(v->getType());
      }
      sumOp = rewriter.create<ONNXReduceSumOp>(loc, types, values, attrs);
    }

    SmallVector<Value, 4> values;
    for (auto v : SmallVector<Value, 4>{sumOp.getODSResults(0)}) {
      values.push_back(v);
    }
    rewriter.replaceOp(op, values);
    return matchSuccess();
  };
};
}  // end anonymous namespace

/// Register optimization patterns as "canonicalization" patterns
/// on the ONNXMatMultOp.
void ONNXAddOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<MulAddToGemmOptPattern>(context);
}
/// on the ONNXIdentityOp.
void ONNXIdentityOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<IdentityEliminationPattern>(context);
}

/// on the ONNXReduceSumSquareOp.
void ONNXReduceSumSquareOp::getCanonicalizationPatterns(
    OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ReduceSumSquareOpPattern>(context);
}
