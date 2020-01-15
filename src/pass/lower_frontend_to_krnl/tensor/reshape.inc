//===----- reshape.inc - Lowering Reshape Op ------------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

struct ONNXReshapeOpLowering : public ConversionPattern {
  ONNXReshapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);
    Value alloc;

    // Compute size in bytes.
    Value tensorSize = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getIntegerType(64),
                                     getMemRefEltSizeInBytes(memRefType)));
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      auto memRefShape = memRefType.getShape();
      SmallVector<Value, 4> allocOperands;
      for (int i = 0; i < memRefShape.size(); ++i) {
        // The shape array can always be used to construct shape information of
        // the result.
        Value index = rewriter.create<ConstantOp>(
            loc, rewriter.getIntegerAttr(rewriter.getIndexType(), i));
        Value loadedVal = rewriter.create<LoadOp>(loc, operands[1], index);
        Value int64LoadedVal = rewriter.create<ZeroExtendIOp>(
            loc, loadedVal, rewriter.getIntegerType(64));
        tensorSize = rewriter.create<MulIOp>(loc, tensorSize, int64LoadedVal);
        allocOperands.push_back(rewriter.create<IndexCastOp>(
            loc, loadedVal, rewriter.getIndexType()));
      }
      AllocOp allocateMemref =
          rewriter.create<AllocOp>(loc, memRefType, allocOperands);

      // Make sure to allocate at the beginning of the block if
      // all dimensions are known.
      auto *parentBlock = allocateMemref.getOperation()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, allocateMemref);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }

      alloc = allocateMemref;
    }

    rewriter.create<KrnlMemcpyOp>(loc, alloc, operands[0], tensorSize);
    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

void populateLoweringONNXReshapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(ctx);
}
