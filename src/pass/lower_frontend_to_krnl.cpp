//====- lower_frontend_to_krnl.cpp - Frontend dialects to Krnl lowering ---===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of frontend operations to a combination of
// Krnl IR and standard operations.
//
//===----------------------------------------------------------------------===//
#include <map>

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"

#include "src/dialect/krnl/krnl_helper.hpp"
#include "src/dialect/krnl/krnl_ops.hpp"
#include "src/dialect/onnx/onnx_ops.hpp"
#include "src/pass/passes.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// FrontendToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Check is all dimensions are known at compile time.
static bool hasAllConstantDimensions(MemRefType type) {
  auto memRefShape = type.getShape();
  for (int i = 0; i < memRefShape.size(); ++i)
    if (memRefShape[i] < 0)
      return false;
  return true;
}

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter,
                                   bool insertDealloc,
                                   ArrayRef<Value> operands = {}) {
  // Put together alloc operands for any dynamic dimensions of the memref.
  AllocOp alloc;
  if (!operands.empty()) {
    auto memRefShape = type.getShape();
    auto rank = memRefShape.size();

    std::map<int, Value> fromOperands;
    for (int reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
      int memRefDimIdx = rank - 1 - reversedIdx;
      if (memRefShape[memRefDimIdx] < 0) { // unknown dimension
        Value maxDim = nullptr;
        for (int i = 0; i < operands.size(); i++) {
          auto operandShape =
              operands[i].getType().cast<MemRefType>().getShape();
          int operandDimIdx = operandShape.size() - 1 - reversedIdx;

          if (operandDimIdx < 0)
            continue;

          // In case of operations with broadcasting, the dimension of the
          // alloc result is the maximum size along each dimension of the
          // operands.
          auto operandDim =
              rewriter.create<DimOp>(loc, operands[i], operandDimIdx);
          if (maxDim) {
            auto maxCondition = rewriter.create<CmpIOp>(loc, CmpIPredicate::sgt,
                                                        operandDim, maxDim);
            maxDim = rewriter.create<SelectOp>(loc, maxCondition, operandDim,
                                               maxDim);
          } else {
            maxDim = operandDim;
          }
        }
        fromOperands.insert(std::make_pair(memRefDimIdx, maxDim));
      }
    }

    SmallVector<Value, 4> allocOperands;
    for (int i = 0; i < rank; ++i)
      if (memRefShape[i] < 0)
        allocOperands.push_back(fromOperands[i]);
    alloc = rewriter.create<AllocOp>(loc, type, allocOperands);
  } else {
    alloc = rewriter.create<AllocOp>(loc, type);
  }

  // Make sure to allocate at the beginning of the block if
  // all dimensions are known.
  auto *parentBlock = alloc.getOperation()->getBlock();
  if (hasAllConstantDimensions(type))
    alloc.getOperation()->moveBefore(&parentBlock->front());

  if (insertDealloc) {
    auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parentBlock->back());
  }

  return alloc;
}

// Determine if current function returns the result value of the
// current op being lowered. If it does then dealloc should not be
// inserted.
static bool checkInsertDealloc(Operation *currentOp) {
  auto parentBlock = currentOp->getBlock();

  bool insertDealloc = true;
  parentBlock->walk([&insertDealloc, currentOp](ReturnOp op) {
    assert(currentOp->getNumResults() < 2 &&
           "No more than one result supported (for now).");
    // If there is at least one result to investigate.
    if (currentOp->getNumResults() > 0) {
      auto result = currentOp->getResult(0);
      for (const auto &operand : op.getOperands())
        if (operand == result)
          insertDealloc = false;
    }
  });

  return insertDealloc;
}

// Create a mapping between an input type's dimensions and a result type's
// dimensions, given that the result type is the result of a reduction op over
// the input type.
std::map<int, int> getReductionMapping(MemRefType inputTy,
                                               ArrayRef<int> axes,
                                               bool keepdims) {
  std::map<int, int> OutInDimMap;
  int64_t rank = inputTy.getRank();

  // Mark reduction axes.
  std::vector<bool> isReductionAxis;
  for (int i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  for (int inIndex = 0, outIndex = 0; inIndex < rank; ++inIndex) {
    // If it is a reduction axis, there is no relationship among dimensions.
    if (isReductionAxis[inIndex]) {
      if (keepdims)
        outIndex++;
    } else {
      OutInDimMap.insert(std::make_pair(outIndex, inIndex));
      outIndex++;
    }
  }

  return OutInDimMap;
}

unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Get run-time dimension information for unknown dimensions used for
// broadcasting.
std::map<int, std::map<int, Value>>
getBroadcastedDimInfo(Location loc, ConversionPatternRewriter &rewriter,
                      MemRefType memRefType, ArrayRef<Value> operands) {
  auto memRefShape = memRefType.getShape();
  int64_t rank = memRefShape.size();
  // For unknown dimensions, we need to get dimension values at runtime in
  // order to do broadcasting.
  std::map<int, std::map<int, Value>> DimInfo;
  // For each result dimension, compute the number of sharing operands.
  // Sharing operands are operands sharing the same index (counting from the
  // rightmost to the leftmost) for a given dimension.
  std::map<int, int> sharedDimCount;
  for (int reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
    int dimIdx = rank - 1 - reversedIdx;
    sharedDimCount[dimIdx] = 0;
    for (int i = 0; i < operands.size(); ++i) {
      auto shape = operands[i].getType().cast<MemRefType>().getShape();
      if (reversedIdx <= shape.size() - 1)
        sharedDimCount[dimIdx]++;
    }
  }
  // An unknown dimension can have a value of 1 or N (N > 1).
  // If its value is 1, it is broadcasted dimension.
  // Otherwise, non-broadcasted dimension.
  // We only care about unknown dimensions whose number of sharing operands is
  // more than one, since they are potentially broadcasted dimensions.
  for (int i = 0; i < operands.size(); ++i) {
    std::map<int, Value> broadcastedDims;
    auto shape = operands[i].getType().cast<MemRefType>().getShape();
    int size = shape.size();
    for (int j = 0; j < shape.size(); ++j) {
      if (shape[j] < 0 and sharedDimCount[rank - size + j] > 1) {
        auto dim = rewriter.create<DimOp>(loc, operands[i], j).getResult();
        auto one = rewriter.create<ConstantIndexOp>(loc, 1);
        auto isBroadcasted =
          rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dim, one);
        broadcastedDims.insert(std::make_pair(j, isBroadcasted));
      }
    }
    DimInfo.insert(std::make_pair(i, broadcastedDims));
  }
  return DimInfo;
}

// Extract induction variables that are used for broadcasting values of a
// given operand.
std::vector<Value>
getLoopIVsForBroadcasting(Location loc, ConversionPatternRewriter &rewriter,
                          ArrayRef<Value> loopIVs, Value operand,
                          std::map<int, Value> broadcastedDims) {
  // `operand` must has a ranked type. This should have been checked by the
  // shape inference pass.
  auto operandShape = operand.getType().cast<MemRefType>().getShape();
  auto rank = operandShape.size();
  auto loopCount = loopIVs.size();

  std::vector<Value> newLoopIVs;
  for (unsigned reversedIdx = 0; reversedIdx < rank; ++reversedIdx) {
    auto dimIdx = rank - 1 - reversedIdx;
    auto loopIdx = loopCount - 1 - reversedIdx;
    if (operandShape[dimIdx] == 1) {
      // Broadcasted dimension
      auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
      newLoopIVs.insert(newLoopIVs.begin(), zero);
    } else if ((operandShape[dimIdx] == -1) &&
               (broadcastedDims.find(dimIdx) != broadcastedDims.end())) {
      // Unknown dimension, it can have a value of 1 or N (N > 1).
      // If its value is 1, it is broadcasted dimension.
      // Otherwise, non-broadcasted dimension.
      auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
      auto idx = rewriter.create<SelectOp>(loc, broadcastedDims[dimIdx], zero,
                                           loopIVs[loopIdx]);
      newLoopIVs.insert(newLoopIVs.begin(), idx);
    } else {
      // Non-broadcasted dimension
      newLoopIVs.insert(newLoopIVs.begin(), loopIVs[loopIdx]);
    }
  }
  return newLoopIVs;
}

namespace {

template <typename ElementwiseNaryOp>
struct ScalarOp;

template <>
struct ScalarOp<ONNXAddOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
struct ScalarOp<ONNXMulOp> {
  using FOp = MulFOp;
  using IOp = MulIOp;
};

template <>
struct ScalarOp<ONNXDivOp> {
  using FOp = DivFOp;
  using IOp = SignedDivIOp;
};

template <>
struct ScalarOp<ONNXSubOp> {
  using FOp = SubFOp;
  using IOp = SubIOp;
};

template <>
struct ScalarOp<ONNXAndOp> {
  using FOp = AndOp; // not use
  using IOp = AndOp;
};

template <>
struct ScalarOp<ONNXOrOp> {
  using FOp = OrOp; // not use
  using IOp = OrOp;
};

template <>
struct ScalarOp<ONNXXorOp> {
  using FOp = XOrOp; // not use
  using IOp = XOrOp;
};

template <>
struct ScalarOp<ONNXExpOp> {
  using FOp = ExpOp;
  using IOp = ExpOp; // not use
};

template <>
struct ScalarOp<ONNXSumOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <>
struct ScalarOp<ONNXTanhOp> {
  using FOp = TanhOp;
  using IOp = TanhOp; // not use
};

template <>
struct ScalarOp<ONNXCosOp> {
  using FOp = CosOp;
  using IOp = CosOp; // not use
};

template <>
struct ScalarOp<ONNXLogOp> {
  using FOp = LogOp;
  using IOp = LogOp; // not use
};

template <>
struct ScalarOp<ONNXReduceProdOp> {
  using FOp = MulFOp;
  using IOp = MulIOp;
};

template <>
struct ScalarOp<ONNXReduceSumOp> {
  using FOp = AddFOp;
  using IOp = AddIOp;
};

template <typename ElementwiseNaryOp>
using ScalarFOp = typename ScalarOp<ElementwiseNaryOp>::FOp;
template <typename ElementwiseNaryOp>
using ScalarIOp = typename ScalarOp<ElementwiseNaryOp>::IOp;

// Get the identity element of a operation.
// Return NULL if the function does not have identity.
template <typename DataType, typename Op>
DataType getIdentityValue() {
  return NULL;
}

template <>
float getIdentityValue<float, ONNXReduceMaxOp>(){
  return (float)-std::numeric_limits<float>::infinity();
}

template <>
int getIdentityValue<int, ONNXReduceMaxOp>(){
  return -std::numeric_limits<int>::infinity();
}

template <>
float getIdentityValue<float, ONNXReduceMinOp>(){
  return (float)std::numeric_limits<float>::infinity();
}

template <>
int getIdentityValue<int, ONNXReduceMinOp>(){
  return std::numeric_limits<int>::infinity();
}

template <>
float getIdentityValue<float, ONNXReduceProdOp>(){
  return (float)1.0;
}

template <>
int getIdentityValue<int, ONNXReduceProdOp>(){
  return 1;
}

template <>
float getIdentityValue<float, ONNXReduceSumOp>(){
  return (float)0;
}

template <>
int getIdentityValue<int, ONNXReduceSumOp>(){
  return 0;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename UnaryOp>
Value mapToLowerScalarOp(Operation *op, ArrayRef<Type> result_types,
                         ArrayRef<Value> operands,
                         ConversionPatternRewriter &rewriter) {
  /* Lower UnaryOp to Ops in the Standard dialect.
   */
  auto loc = op->getLoc();
  Type element_type = operands.front().getType();
  if (element_type.isa<IntegerType>()) {
    return rewriter.create<ScalarIOp<UnaryOp>>(loc, result_types, operands,
                                               mlir::None);
  } else if (element_type.isa<FloatType>()) {
    return rewriter.create<ScalarFOp<UnaryOp>>(loc, result_types, operands,
                                               mlir::None);
  } else {
    emitError(loc, "unsupported element type");
    return nullptr;
  }
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSinhOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSinhOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXSinhOp(%X) = DivFOp(SubFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto two = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 2));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXCoshOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXCoshOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXCoshOp(%X) = DivFOp(AddFOp(ExpOp(%X), ExpOp(NegFOp(%X))),
  //                         ConstantOp 2)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto two = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 2));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSigmoidOp>(Operation *op,
                                        ArrayRef<Type> result_types,
                                        ArrayRef<Value> operands,
                                        ConversionPatternRewriter &rewriter) {
  // ONNXSigmoidOp(%X) = DivFOp(ConstantOp 1,
  //                            AddFOp(ConstantOp 1, ExpOp(NegFOp(%X))))
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto neg = rewriter.create<SubFOp>(loc, zero, operand);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, one, rewriter.create<AddFOp>(loc, one, negExp));

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXHardSigmoidOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXHardSigmoidOp>(
    Operation *op, ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // %Y = AddFOp(MulFOp(alpha, %X), beta)
  // %Z = SelectOp(CmpFOp(OGT, %Y, Constant 0),
  //               %Y,
  //               Constant 0)
  // ONNXHardSigmoidOp(%X) = SelectOp(CmpFOp(OLT, %Z, Constant 1),
  //                                  %Z,
  //                                  Constant 1)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto alphaAttr = op->getAttrOfType<FloatAttr>("HardSigmoid.alpha");
  auto betaAttr = op->getAttrOfType<FloatAttr>("HardSigmoid.beta");
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttr);
  auto beta = rewriter.create<ConstantOp>(loc, betaAttr);

  auto add = rewriter.create<AddFOp>(
      loc, rewriter.create<MulFOp>(loc, alpha, operand), beta);
  auto maxPredicate =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, add, zero);
  auto max = rewriter.create<SelectOp>(loc, maxPredicate, add, zero);
  auto minPredicate =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, max, one);
  auto result = rewriter.create<SelectOp>(loc, minPredicate, max, one);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXEluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXEluOp>(Operation *op, ArrayRef<Type> result_types,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  // ONNXEluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                          MulFOp(alpha, SubFOp(ExpOp(%X), 1)),
  //                          %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto alphaAttr = op->getAttrOfType<FloatAttr>("Elu.alpha");
  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttr);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(
      loc, lessThanZero,
      rewriter.create<MulFOp>(loc, alpha,
                              rewriter.create<SubFOp>(loc, exp, one)),
      operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReluOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                           ConstantOp 0,
  //                           %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(loc, lessThanZero, zero, operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXLeakyReluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXLeakyReluOp>(Operation *op,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> operands,
                                          ConversionPatternRewriter &rewriter) {
  // ONNXLeakyReluOp(%X) = SelectOp(CmpFOp(OLT, %X, ConstantOp 0),
  //                                MulFOp(alpha, %X),
  //                                %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto alphaAttr = op->getAttrOfType<FloatAttr>("LeakyRelu.alpha");
  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttr);
  auto lessThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, operand, zero);
  auto result = rewriter.create<SelectOp>(
      loc, lessThanZero, rewriter.create<MulFOp>(loc, alpha, operand), operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXSeluOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXSeluOp>(Operation *op, ArrayRef<Type> result_types,
                                     ArrayRef<Value> operands,
                                     ConversionPatternRewriter &rewriter) {
  // ONNXSeluOp(%X) = SelectOp(CmpFOp(OGT, %X, ConstantOp 0),
  //                           MulFOp(gamma, %X),
  //                           MulFOp(gamma,
  //                                  SubFOp(MulFOp(alpha, ExpOp(%X)),
  //                                         alpha)))
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto alphaAttr = op->getAttrOfType<FloatAttr>("Selu.alpha");
  auto gammaAttr = op->getAttrOfType<FloatAttr>("Selu.gamma");
  auto elementType = result_types[0];

  auto zero = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 0));
  auto alpha = rewriter.create<ConstantOp>(loc, alphaAttr);
  auto gamma = rewriter.create<ConstantOp>(loc, gammaAttr);
  auto exp = rewriter.create<ExpOp>(loc, operand);
  auto greaterThanZero =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, operand, zero);
  auto select = rewriter.create<SelectOp>(
      loc, greaterThanZero, operand,
      rewriter.create<SubFOp>(loc, rewriter.create<MulFOp>(loc, alpha, exp),
                              alpha));
  auto result = rewriter.create<MulFOp>(loc, gamma, select);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReciprocalOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReciprocalOp>(
    Operation *op, ArrayRef<Type> result_types, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) {
  // ONNXReciprocalOp(%X) = DivFOp(ConstantOp 1, %X)
  auto loc = op->getLoc();
  Value operand = operands[0];
  auto elementType = result_types[0];

  auto one = rewriter.create<ConstantOp>(loc, FloatAttr::get(elementType, 1));
  auto result = rewriter.create<DivFOp>(loc, one, operand);

  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMaxOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXMaxOp>(Operation *op, ArrayRef<Type> result_types,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  // ONNXMaxOp(%X, %Y) = SelectOp(CmpFOp(OGT, %X, %Y),
  //                              %X,
  //                              %Y)
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXMinOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXMinOp>(Operation *op, ArrayRef<Type> result_types,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) {
  // ONNXMinOp(%X, %Y) = SelectOp(CmpFOp(OLT, %X, %Y),
  //                              %X,
  //                              %Y)
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto min = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMaxOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReduceMaxOp>(Operation *op,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> operands,
                                          ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto max = rewriter.create<CmpFOp>(loc, CmpFPredicate::OGT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, max, lhs, rhs);
  return result;
}

//===----------------------------------------------------------------------===//
// Scalar unary ops for lowering ONNXReduceMinOp
//===----------------------------------------------------------------------===//
template <>
Value mapToLowerScalarOp<ONNXReduceMinOp>(Operation *op,
                                          ArrayRef<Type> result_types,
                                          ArrayRef<Value> operands,
                                          ConversionPatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value lhs = operands[0];
  Value rhs = operands[1];
  auto min = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, lhs, rhs);
  auto result = rewriter.create<SelectOp>(loc, min, lhs, rhs);
  return result;
}

// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering : public ConversionPattern {
  ONNXElementwiseUnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseUnaryOp::getOperationName(), 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: Check that the types are valid.
    // An element-wise unary operation must have all operands and the result of
    // the same type. This should have been verified by the verifier.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);

    // If the output has a dynamic dimension, pass the operands required for
    // each dynamic dimension to the AllocOp. The first operand of the
    // operation is used. The operands of the op need to match in terms of
    // dimensions with the result at this pre-optimization phase.
    // TODO: verify that dimensions match.
    // TODO: can the dimension of the result differ after optimizations?
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    {operands[0]});

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Define loops.
    auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, rank);
    std::vector<Value> originalLoops;
    originalLoops.reserve(rank);
    for (auto result : loopsOp.getResults()) {
      originalLoops.push_back(result);
    }

    // Define loop optimization.
    auto optimizedLoopsOp = rewriter.create<KrnlOptimizeLoopsOp>(loc, rank);
    std::vector<Value> optimizedLoops;
    optimizedLoops.reserve(rank);
    for (auto result : optimizedLoopsOp.getResults()) {
      optimizedLoops.push_back(result);
    }
    Block &optimizationBlock = optimizedLoopsOp.region().front();

    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    // Iterate over the loop nest.
    // TODO (Tian): move this logic inside KrnlIterateOp. Pass MemRefShape
    // to KrnlIterateOp instead.
    for (int i = 0; i < rank; ++i) {
      if (memRefShape[i] < 0) {
        pack.pushConstantBound(0);
        pack.pushOperandBound(
            rewriter.create<DimOp>(loc, operands[0], i).getResult());
      } else {
        pack.pushConstantBound(0);
        pack.pushConstantBound(memRefShape[i]);
      }
    }

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(&optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops
    // unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    rewriter.setInsertionPoint(optimizedLoopsOp);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlock.getArguments())
      loopIVs.push_back(arg);

    auto loadedVal = rewriter.create<LoadOp>(loc, operands[0], loopIVs);
    auto loweredOpResult = mapToLowerScalarOp<ElementwiseUnaryOp>(
        op, memRefType.getElementType(), {loadedVal}, rewriter);
    // Store result in the resulting array.
    rewriter.create<StoreOp>(loc, loweredOpResult, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering : public ConversionPattern {
  ONNXElementwiseVariadicOpLowering(MLIRContext *ctx)
      : ConversionPattern(ElementwiseVariadicOp::getOperationName(), 1, ctx) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // TODO: Check that the types are valid.
    // An element-wise variadic operation must have all operands and the result
    // of the same type. This should have been verified by the verifier.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto loc = op->getLoc();
    auto numArgs = op->getNumOperands();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertTensorToMemRef(tensorType);

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    // If the output has a dynamic dimension, we compute its dimension at
    // runtime by using dimensions from the operands.
    // In particular, we need to know from which operand a result dimension
    // comes from.
    // TODO: can the dimension of the result differ after optimizations?
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc,
                                    operands);

    // Number of loops
    auto memRefShape = memRefType.getShape();
    int64_t rank = memRefShape.size();

    // Define loops.
    auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, rank);
    std::vector<Value> originalLoops;
    originalLoops.reserve(rank);
    for (auto result : loopsOp.getResults()) {
      originalLoops.push_back(result);
    }

    // Define loop optimization.
    auto optimizedLoopsOp = rewriter.create<KrnlOptimizeLoopsOp>(loc, rank);
    std::vector<Value> optimizedLoops;
    optimizedLoops.reserve(rank);
    for (auto result : optimizedLoopsOp.getResults()) {
      optimizedLoops.push_back(result);
    }
    Block &optimizationBlock = optimizedLoopsOp.region().front();

    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    // Iterate over the loop nest.
    // TODO (Tian): move this logic inside KrnlIterateOp. Pass MemRefShape
    // to KrnlIterateOp instead.
    for (int i = 0; i < rank; ++i) {
      if (memRefShape[i] < 0) {
        pack.pushConstantBound(0);
        pack.pushOperandBound(
            rewriter.create<DimOp>(loc, alloc, i).getResult());
      } else {
        pack.pushConstantBound(0);
        pack.pushConstantBound(memRefShape[i]);
      }
    }

    // Get run-time dimension information for unknown dimensions used for
    // broadcasting.
    std::map<int, std::map<int, Value>> broadcastedDimInfo =
        getBroadcastedDimInfo(loc, rewriter, memRefType, operands);

    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Now perform the insertions into the body of the
    // just generated instructions:

    // 1. Insert any optimizations in the KrnlOptimizeLoopsOp body.
    rewriter.setInsertionPointToEnd(&optimizationBlock);
    // Return from KrnlOptimizeLoopsOp body.
    // When no optimizations are present we just return the loops unchaged.
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    rewriter.setInsertionPoint(optimizedLoopsOp);

    // 2. Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlock.getArguments())
      loopIVs.push_back(arg);

    // Fold over operands for each of their scalar values
    Value accumulated, next;
    auto accumulatedLoopIVs = getLoopIVsForBroadcasting(
        loc, rewriter, loopIVs, operands[0], broadcastedDimInfo[0]);
    accumulated = rewriter.create<LoadOp>(loc, operands[0], accumulatedLoopIVs);
    for (unsigned i = 1; i < numArgs; i++) {
      auto nextLoopIVs = getLoopIVsForBroadcasting(
          loc, rewriter, loopIVs, operands[i], broadcastedDimInfo[i]);
      next = rewriter.create<LoadOp>(loc, operands[i], nextLoopIVs);
      accumulated = mapToLowerScalarOp<ElementwiseVariadicOp>(
          op, memRefType.getElementType(), {accumulated, next}, rewriter);
    }
    // Store result in the resulting array.
    rewriter.create<StoreOp>(loc, accumulated, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);

    return matchSuccess();
  }
};

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

//===----------------------------------------------------------------------===//
// Reduction ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//
template <typename ONNXReductionOp>
struct ONNXReductionOpLowering : public ConversionPattern {
  ONNXReductionOpLowering(MLIRContext *ctx)
      : ConversionPattern(ONNXReductionOp::getOperationName(), 1, ctx) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    /*
     * Condition: reduction function must be associative and communicative.
     *
     * Example 1 (here, reduction function is `+`):
     * Induction variables: (i0, i1, i2)
     * axes = [0, 2]
     * keepdims = true
     * krnl.iterate() with (i0, i1, i2) {
     *   Y(0, i1, 0) += X(i0, i1, i2)
     * }
     *
     * Example 2 (here, reduction function is `+`):
     * Induction variables: (i0, i1, i2)
     * axes = [0, 2]
     * keepdims = false
     * krnl.iterate() with (i0, i1, i2) {
     *   Y(i1) += X(i0, i1, i2)
     * }
     *
    */
    auto loc = op->getLoc();
    auto memRefInType = operands[0].getType().cast<MemRefType>();
    auto memRefInShape = memRefInType.getShape();
    auto tensorOutType = (*op->result_type_begin()).cast<TensorType>();
    int64_t inRank = memRefInType.getRank();
    int64_t outRank = tensorOutType.getRank();

    // Get attributes
    auto axisAttrs = op->getAttrOfType<ArrayAttr>("axes");
    std::vector<int> axes;
    if (axisAttrs) {
      for (auto axisAttr : axisAttrs.getValue()) {
        int axis = axisAttr.cast<IntegerAttr>().getInt();
        axis = axis >= 0 ? axis : (outRank + axis);
        assert(axis >= -outRank && axis <= outRank - 1);
        if (std::find(axes.begin(), axes.end(), axis) == axes.end())
          axes.push_back(axis);
      }
    } else {
      for (int i = 0; i < outRank; ++i) {
        axes.push_back(i);
      }
    }
    // KeepDims
    int keepdimsAttr = op->getAttrOfType<IntegerAttr>("keepdims").getInt();
    bool isKeepdims = (keepdimsAttr == 1) ? true : false;

    // Get type information
    auto memRefOutType = convertTensorToMemRef(tensorOutType);
    auto memRefOutShape = memRefOutType.getShape();
    auto elementOutType = memRefOutType.getElementType();
    std::map<int, int> outInDimMap =
        getReductionMapping(memRefInType, axes, isKeepdims);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefOutType)) {
      alloc = insertAllocAndDealloc(memRefOutType, loc, rewriter, insertDealloc);
    } else {
      SmallVector<Value, 2> allocOperands;
      for (int i = 0; i < outRank; ++i) {
        if (memRefOutShape[i] < 0) {
          auto dim = rewriter.create<DimOp>(loc, operands[0], outInDimMap[i]);
          allocOperands.push_back(dim);
        }
      }
      alloc = rewriter.create<AllocOp>(loc, memRefOutType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }

    // There are two Krnl loops:
    // - One to initialize the result memref, and
    // - One to do reduction

    // Define an loop to initialize the result.
    auto loopsOpInit = rewriter.create<KrnlDefineLoopsOp>(loc, outRank);
    std::vector<Value> originalLoopsInit;
    originalLoopsInit.reserve(outRank);
    for (auto result : loopsOpInit.getResults()) {
      originalLoopsInit.push_back(result);
    }
    // Loop optimization.
    auto optimizedLoopsOpInit =
        rewriter.create<KrnlOptimizeLoopsOp>(loc, outRank);
    std::vector<Value> optimizedLoopsInit;
    optimizedLoopsInit.reserve(outRank);
    for (auto result : optimizedLoopsOpInit.getResults()) {
      optimizedLoopsInit.push_back(result);
    }
    Block &optimizationBlockInit = optimizedLoopsOpInit.region().front();
    // Iteration information
    KrnlIterateOperandPack packInit(rewriter, originalLoopsInit,
        optimizedLoopsInit);
    for (int i = 0; i < outRank; ++i) {
      if (memRefOutShape[i] < 0) {
        packInit.pushConstantBound(0);
        packInit.pushOperandBound(
            rewriter.create<DimOp>(loc, alloc, i).getResult());
      } else {
        packInit.pushConstantBound(0);
        packInit.pushConstantBound(memRefOutShape[i]);
      }
    }
    auto iterateOpInit = rewriter.create<KrnlIterateOp>(loc, packInit);
    Block &iterationBlockInit = iterateOpInit.bodyRegion().front();

    // Perform the insertions into the body of the initialization loop.
    // No optimization
    rewriter.setInsertionPointToEnd(&optimizationBlockInit);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoopsInit);
    rewriter.setInsertionPoint(optimizedLoopsOpInit);

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlockInit);

    // Handle the operation:
    SmallVector<Value, 4> loopIVs;
    for (auto arg : iterationBlockInit.getArguments()) {
      loopIVs.push_back(arg);
    }

    Value identity;
    if (elementOutType.isa<FloatType>()) {
      identity = rewriter.create<ConstantOp>(
          loc, FloatAttr::get(elementOutType,
                              getIdentityValue<float, ONNXReductionOp>()));
    } else if (elementOutType.isa<IntegerType>()) {
      identity = rewriter.create<ConstantOp>(
          loc, IntegerAttr::get(elementOutType,
                                getIdentityValue<int, ONNXReductionOp>()));
    } else {
      emitError(loc, "unsupported element type");
    }
    rewriter.create<StoreOp>(loc, identity, alloc, loopIVs);

    // Define an Krnl loop to do reduction.
    rewriter.setInsertionPointAfter(iterateOpInit);
    auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, inRank);
    std::vector<Value> originalLoops;
    originalLoops.reserve(inRank);
    for (auto result : loopsOp.getResults()) {
      originalLoops.push_back(result);
    }
    // Loop optimization.
    auto optimizedLoopsOp = rewriter.create<KrnlOptimizeLoopsOp>(loc, inRank);
    std::vector<Value> optimizedLoops;
    optimizedLoops.reserve(inRank);
    for (auto result : optimizedLoopsOp.getResults()) {
      optimizedLoops.push_back(result);
    }
    Block &optimizationBlock = optimizedLoopsOp.region().front();
    // Iteration information
    KrnlIterateOperandPack pack(rewriter, originalLoops, optimizedLoops);
    for (int i = 0; i < inRank; ++i) {
      if (memRefInShape[i] < 0) {
        pack.pushConstantBound(0);
        pack.pushOperandBound(
            rewriter.create<DimOp>(loc, operands[0], i).getResult());
      } else {
        pack.pushConstantBound(0);
        pack.pushConstantBound(memRefInShape[i]);
      }
    }
    auto iterateOp = rewriter.create<KrnlIterateOp>(loc, pack);
    Block &iterationBlock = iterateOp.bodyRegion().front();

    // Perform the insertions into the body of the reduction loop.
    // No optimization
    rewriter.setInsertionPointToEnd(&optimizationBlock);
    rewriter.create<KrnlReturnLoopsOp>(loc, originalLoops);
    rewriter.setInsertionPoint(optimizedLoopsOp);

    // Insert instructions inside the KernelIterateOp body.
    rewriter.setInsertionPointToStart(&iterationBlock);

    // Handle the operation:
    SmallVector<Value, 4> inLoopIVs, outLoopIVs;
    auto args = iterationBlock.getArguments();
    for (int i = 0; i < args.size(); ++i) {
      inLoopIVs.push_back(args[i]);
    }
    Value zeroIndex = nullptr;
    for (int i = 0; i < outRank; ++i) {
      if (outInDimMap.find(i) != outInDimMap.end()) {
        outLoopIVs.push_back(inLoopIVs[outInDimMap[i]]);
      } else {
        if (zeroIndex) {
          outLoopIVs.push_back(zeroIndex);
        } else {
          zeroIndex = rewriter.create<ConstantIndexOp>(loc, 0);
          outLoopIVs.push_back(zeroIndex);
        }
      }
    }

    Value next, accumulated;
    next = rewriter.create<LoadOp>(loc, operands[0], inLoopIVs);
    accumulated = rewriter.create<LoadOp>(loc, alloc, outLoopIVs);
    accumulated = mapToLowerScalarOp<ONNXReductionOp>(
        op, memRefOutType.getElementType(), {accumulated, next}, rewriter);
    rewriter.create<StoreOp>(loc, accumulated, alloc, outLoopIVs);

    rewriter.replaceOp(op, alloc);
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// EntryPoint Op lowering to Krnl Entry Point.
//===----------------------------------------------------------------------===//

class ONNXEntryPointLowering : public OpRewritePattern<ONNXEntryPointOp> {
public:
  using OpRewritePattern<ONNXEntryPointOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ONNXEntryPointOp op,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<KrnlEntryPointOp>(
        op,
        op.getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName()),
        op.getAttrOfType<IntegerAttr>(ONNXEntryPointOp::getNumInputsAttrName()),
        op.getAttrOfType<IntegerAttr>(
            ONNXEntryPointOp::getNumOutputsAttrName()));
    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Conversion from Tensor type to the Standard dialect MemRef type.
//===----------------------------------------------------------------------===//

struct TensorTypeConverter : public TypeConverter {
  using TypeConverter::TypeConverter;

  LogicalResult convertType(Type t, SmallVectorImpl<Type> &results) override {
    if (auto tensor_type = t.dyn_cast<TensorType>()) {
      results.push_back(convertTensorToMemRef(tensor_type));
      return success();
    }

    results.push_back(t);
    return success();
  }

  /// Return true if the inputs and outputs of the given function type are
  /// legal. [Taken from MLIR and adapted to only check the legality of the
  /// inputs. Once unranked results can be handled gracefully this
  /// override needs to be removed in favour of the original MLIR one.]
  bool isSignatureLegal(FunctionType funcType) {
    return llvm::all_of(funcType.getInputs(),
                        [this](Type type) { return isLegal(type); });
  }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// Frontend to Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to Krnl loops of the ONNX operations.
namespace {
struct FrontendToKrnlLoweringPass
    : public ModulePass<FrontendToKrnlLoweringPass> {
  void runOnModule() final;
};
} // end anonymous namespace.

void FrontendToKrnlLoweringPass::runOnModule() {
  auto module = getModule();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target
      .addLegalDialect<KrnlOpsDialect, AffineOpsDialect, StandardOpsDialect>();

  // TODO: enable this once more ops are supported.
  // We also define the ONNX dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted.
  // target.addIllegalDialect<mlir::ONNXOpsDialect>();

  // TODO: add any other ops which are considered legal.
  // Some operations can be marked as being still legal.
  // Example: target.addLegalOp<mlir::OpName>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the frontend operations.
  OwningRewritePatternList patterns;

  // Convert TensorType to MemRef
  TensorTypeConverter tensor_to_memref_converter;
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    // FuncOp is legal only if types have been converted to Std types.
    return tensor_to_memref_converter.isSignatureLegal(op.getType());
  });

  // Type conversion for function signatures.
  // Call MLIR FuncOp signature conversion when result type is
  // a ranked tensor.
  populateFuncOpTypeConversionPattern(patterns, &getContext(),
                                      tensor_to_memref_converter);

  // Frontent operation lowering.
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXTanhOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXLogOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXHardSigmoidOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXEluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXLeakyReluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXSeluOp>,
                  ONNXElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXAddOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXMulOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXDivOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXSubOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXAndOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXOrOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXXorOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXSumOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXMaxOp>,
                  ONNXElementwiseVariadicOpLowering<mlir::ONNXMinOp>,
                  ONNXReshapeOpLowering, ONNXEntryPointLowering,
                  ONNXReductionOpLowering<mlir::ONNXReduceMaxOp>,
                  ONNXReductionOpLowering<mlir::ONNXReduceMinOp>,
                  ONNXReductionOpLowering<mlir::ONNXReduceProdOp>,
                  ONNXReductionOpLowering<mlir::ONNXReduceSumOp>>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, patterns)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToKrnlPass() {
  return std::make_unique<FrontendToKrnlLoweringPass>();
}

static PassRegistration<FrontendToKrnlLoweringPass>
    pass("lower-frontend", "Lower frontend ops to Krnl dialect.");
