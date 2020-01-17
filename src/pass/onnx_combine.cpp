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

// ONNXReduceSumSquareOp %X = ONNXReduceSumOp (ONNXMulOp %X, %X)
struct ReduceSumSquareOpPattern : public RewritePattern {
  ReduceSumSquareOpPattern(MLIRContext *context)
      : RewritePattern("onnx.ReduceSumSquare", {"onnx.Mul", "onnx.ReduceSum"},
                       1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used for creating ops
    Operation::operand_range X(op0->getOperands());
    Operation *ops[1];

    // Match
    ops[0] = op0;
    auto sumSquareOp = dyn_cast_or_null<ONNXReduceSumSquareOp>(op0);
    X = sumSquareOp.getODSOperands(0);

    // Rewrite
    auto loc = rewriter.getFusedLoc({ops[0]->getLoc()});
    ONNXMulOp mulOp;
    {
      Value A = (*X.begin());
      Value B = (*X.begin());
      SmallVector<Type, 4> outTypes;
      for (auto v : sumSquareOp.getODSResults(0)) {
        outTypes.push_back(v->getType());
      }
      mulOp = rewriter.create<ONNXMulOp>(loc, outTypes, A, B);
    }

    ONNXReduceSumOp sumOp;
    {
      SmallVector<Value, 4> values;
      SmallVector<NamedAttribute, 4> attrs;
      SmallVector<Type, 4> outTypes;
      values.push_back((*mulOp.getODSResults(0).begin()));
      for (auto attr : sumSquareOp.getAttrs()) {
        attrs.push_back(attr);
      }
      for (auto v : sumSquareOp.getODSResults(0)) {
        outTypes.push_back(v->getType());
      }
      sumOp = rewriter.create<ONNXReduceSumOp>(loc, outTypes, values, attrs);
    }

    SmallVector<Value, 4> values;
    for (auto v : SmallVector<Value, 4>{sumOp.getODSResults(0)}) {
      values.push_back(v);
    }
    rewriter.replaceOp(op0, values);
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
