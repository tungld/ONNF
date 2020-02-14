//===- onnx_rewrite.cpp - ONNX High Level Optimizer -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of rewriters for operations in the ONNX dialect
// that can be rewritten by using other ONNX operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "src/dialect/onnx/onnx_ops.hpp"

using namespace mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/onnx_rewrite.inc"

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
struct ReduceL2OpPattern : public RewritePattern {
  ReduceL2OpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceL2Op::getOperationName(),
                       {ONNXSqrtOp::getOperationName(),
                        ONNXReduceSumSquareOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXReduceSumSquareOp sumSquareOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      sumSquareOp = rewriter.create<ONNXReduceSumSquareOp>(
          loc, UnrankedTensorType::get(elementType), opInput, opAttrs);
    }

    ONNXSqrtOp sqrtOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }
      sqrtOp = rewriter.create<ONNXSqrtOp>(loc, types, sumSquareOp.getResult());
    }

    rewriter.replaceOp(op, sqrtOp.getResult());
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
struct ReduceLogSumOpPattern : public RewritePattern {
  ReduceLogSumOpPattern(MLIRContext *context)
      : RewritePattern(ONNXReduceLogSumOp::getOperationName(),
                       {ONNXReduceSumOp::getOperationName(),
                        ONNXLogOp::getOperationName()},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto opInput = op->getOperands()[0]; // %X
    auto opResults = op->getResults();
    auto opAttrs = op->getAttrs();

    // Rewrite
    ONNXReduceSumOp sumOp;
    {
      auto elementType = opInput.getType().cast<TensorType>().getElementType();
      sumOp = rewriter.create<ONNXReduceSumOp>(
          loc, UnrankedTensorType::get(elementType), opInput, opAttrs);
    }

    ONNXLogOp logOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opResults) {
        types.emplace_back(v.getType());
      }
      logOp = rewriter.create<ONNXLogOp>(loc, types, sumOp.getResult());
    }

    rewriter.replaceOp(op, logOp.getResult());
    return matchSuccess();
  };
};

} // end anonymous namespace

/// on the ONNXReduceL1Op.
void ONNXReduceL1Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceL1OpPattern>(context);
}
/// on the ONNXReduceL2Op.
void ONNXReduceL2Op::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceL2OpPattern>(context);
}

/// on the ONNXReduceLogSumOp.
void ONNXReduceLogSumOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceLogSumOpPattern>(context);
}

/// on the ONNXReduceLogSumExpOp.
void ONNXReduceLogSumExpOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceLogSumExpOpPattern>(context);
}

/// on the ONNXReduceSumSquareOp.
void ONNXReduceSumSquareOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<ReduceSumSquareOpPattern>(context);
}
