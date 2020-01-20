//===- onnx_rewrite.cpp - ONNX High Level Optimizer ------------------------===//
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

// Due to a limitation of ODS that "all ODS-generated `build()` methods require
// specifying the result type(s), unless the op has known traits like
// `SameOperandsAndResultType` that we can use to auto-generate a `build()`
// method with result type deduction", we will to rewrite some patterns manually
// until we find a better way.
//
// More information about the limitation can be found here:
// https://github.com/llvm/llvm-project/blob/master/mlir/docs/DeclarativeRewrites.md#building-operations
//

//===----------------------------------------------------------------------===//
// ONNXReduceL1Op %X = ONNXReduceSumOp (ONNXAbsOp %X)
//===----------------------------------------------------------------------===//
struct ReduceL1OpPattern : public RewritePattern {
  ReduceL1OpPattern(MLIRContext *context)
      : RewritePattern("onnx.ReduceL1", {"onnx.Abs", "onnx.ReduceSum"}, 1,
                       context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range X(op->getOperands());

    // Match
    auto opODS = dyn_cast_or_null<ONNXReduceL1Op>(op);
    auto opODS_X = opODS.getODSOperands(0);
    auto opODS_Y = opODS.getODSResults(0);
    auto opODS_Attrs = opODS.getAttrs();

    // Rewrite
    auto loc = op->getLoc();
    ONNXAbsOp absOp;
    {
      Value X = (*opODS_X.begin());
      auto elementType = X->getType().cast<TensorType>().getElementType();
      absOp = rewriter.create<ONNXAbsOp>(
          loc, UnrankedTensorType::get(elementType), X);
    }

    ONNXReduceSumOp sumOp;
    {
      SmallVector<Value, 4> values;
      SmallVector<NamedAttribute, 4> attrs;
      SmallVector<Type, 4> types;
      values.push_back((*absOp.getODSResults(0).begin()));
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

//===----------------------------------------------------------------------===//
// ONNXReduceL2Op %X = ONNXSqrtOp (ONNXReduceSumSquareOp (%X))
//===----------------------------------------------------------------------===//
struct ReduceL2OpPattern : public RewritePattern {
  ReduceL2OpPattern(MLIRContext *context)
      : RewritePattern("onnx.ReduceL2", {"onnx.Sqrt", "onnx.ReduceSumSquare"},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range X(op->getOperands());

    // Match
    auto opODS = dyn_cast_or_null<ONNXReduceL2Op>(op);
    auto opODS_X = opODS.getODSOperands(0);
    auto opODS_Y = opODS.getODSResults(0);
    auto opODS_Attrs = opODS.getAttrs();

    // Rewrite
    auto loc = op->getLoc();
    ONNXReduceSumSquareOp sumSquareOp;
    {
      Value X = (*opODS_X.begin());
      auto elementType = X->getType().cast<TensorType>().getElementType();
      sumSquareOp = rewriter.create<ONNXReduceSumSquareOp>(
          loc, UnrankedTensorType::get(elementType), X, opODS_Attrs);
    }

    ONNXSqrtOp sqrtOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opODS_Y) {
        types.push_back(v->getType());
      }
      sqrtOp = rewriter.create<ONNXSqrtOp>(
          loc, types, (*sumSquareOp.getODSResults(0).begin()));
    }

    SmallVector<Value, 4> values;
    for (auto v : SmallVector<Value, 4>{sqrtOp.getODSResults(0)}) {
      values.push_back(v);
    }
    rewriter.replaceOp(op, values);
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumOp %X = ONNXLogOp (ONNXReduceSumOp (%X))
//===----------------------------------------------------------------------===//
struct ReduceLogSumOpPattern : public RewritePattern {
  ReduceLogSumOpPattern(MLIRContext *context)
      : RewritePattern("onnx.ReduceLogSum", {"onnx.ReduceSum", "onnx.Log"}, 1,
                       context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range X(op->getOperands());

    // Match
    auto opODS = dyn_cast_or_null<ONNXReduceLogSumOp>(op);
    auto opODS_X = opODS.getODSOperands(0);
    auto opODS_Y = opODS.getODSResults(0);
    auto opODS_Attrs = opODS.getAttrs();

    // Rewrite
    auto loc = op->getLoc();
    ONNXReduceSumOp sumOp;
    {
      Value X = (*opODS_X.begin());
      auto elementType = X->getType().cast<TensorType>().getElementType();
      sumOp = rewriter.create<ONNXReduceSumOp>(
          loc, UnrankedTensorType::get(elementType), X, opODS_Attrs);
    }

    ONNXLogOp logOp;
    {
      SmallVector<Type, 4> types;
      for (auto v : opODS_Y) {
        types.push_back(v->getType());
      }
      logOp = rewriter.create<ONNXLogOp>(loc, types,
                                         (*sumOp.getODSResults(0).begin()));
    }

    SmallVector<Value, 4> values;
    for (auto v : SmallVector<Value, 4>{logOp.getODSResults(0)}) {
      values.push_back(v);
    }
    rewriter.replaceOp(op, values);
    return matchSuccess();
  };
};

//===----------------------------------------------------------------------===//
// ONNXReduceLogSumExpOp %X = ONNXReduceLogSumOp (ONNXExpOp %X)
//===----------------------------------------------------------------------===//
struct ReduceLogSumExpOpPattern : public RewritePattern {
  ReduceLogSumExpOpPattern(MLIRContext *context)
      : RewritePattern("onnx.ReduceLogSumExp",
                       {"onnx.Exp", "onnx.ReduceLogSum"}, 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range X(op->getOperands());

    // Match
    auto opODS = dyn_cast_or_null<ONNXReduceLogSumExpOp>(op);
    auto opODS_X = opODS.getODSOperands(0);
    auto opODS_Y = opODS.getODSResults(0);
    auto opODS_Attrs = opODS.getAttrs();

    // Rewrite
    auto loc = op->getLoc();
    ONNXExpOp expOp;
    {
      Value X = (*opODS_X.begin());
      auto elementType = X->getType().cast<TensorType>().getElementType();
      expOp = rewriter.create<ONNXExpOp>(
          loc, UnrankedTensorType::get(elementType), X);
    }

    ONNXReduceLogSumOp logSumOp;
    {
      SmallVector<Value, 4> values;
      SmallVector<NamedAttribute, 4> attrs;
      SmallVector<Type, 4> types;
      values.push_back((*expOp.getODSResults(0).begin()));
      for (auto attr : opODS_Attrs) {
        attrs.push_back(attr);
      }
      for (auto v : opODS_Y) {
        types.push_back(v->getType());
      }
      logSumOp = rewriter.create<ONNXReduceLogSumOp>(loc, types, values, attrs);
    }

    SmallVector<Value, 4> values;
    for (auto v : SmallVector<Value, 4>{logSumOp.getODSResults(0)}) {
      values.push_back(v);
    }
    rewriter.replaceOp(op, values);
    return matchSuccess();
  };
};

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
