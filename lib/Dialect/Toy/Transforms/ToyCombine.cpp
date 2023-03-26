//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Toy dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Toy/IR/ToyDialect.h"
#include "Dialect/Toy/IR/ToyOps.h"
#include "Dialect/Toy/IR/ToyTypes.h"

#include "EGraph/EGraph.h"
#include "EGraph/OpEGraphRewritePattern.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include <numeric>
using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose
    : public mlir::OpEGraphRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    mlir::Value transposeInput = op->getOperand(0);
    Operation *inputOp = transposeInput.getDefiningOp();

    if (TransposeOp transposeInputOp =
            eGraph.getOpInEClass<TransposeOp>(inputOp)) {
      mlir::Value ret = transposeInputOp.getOperand();
      return ret.getDefiningOp();
    }
    return nullptr;
  }
};

struct TransposeAddition : public mlir::OpEGraphRewritePattern<TransposeOp> {
  TransposeAddition(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    mlir::Value transposeInput = op->getOperand(0);
    Operation *inputOp = transposeInput.getDefiningOp();

    if (AddOp transposeInputOp = eGraph.getOpInEClass<AddOp>(inputOp)) {
      Value TransposeValue0 =
          rewriter
              .create<TransposeOp>(op->getLoc(), transposeInputOp.getOperand(0))
              .getResult();
      Value TransposeValue1 =
          rewriter
              .create<TransposeOp>(op->getLoc(), transposeInputOp.getOperand(1))
              .getResult();
      return rewriter.create<AddOp>(op->getLoc(), TransposeValue0,
                                    TransposeValue1);
    }
    return nullptr;
  }
};

struct AdditionTranspose : public mlir::OpEGraphRewritePattern<AddOp> {
  AdditionTranspose(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, /*benefit=*/1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    mlir::Value addInput0 = op->getOperand(0);
    Operation *inputOp0 = addInput0.getDefiningOp();
    mlir::Value addInput1 = op->getOperand(1);
    Operation *inputOp1 = addInput1.getDefiningOp();

    TransposeOp addInputOp0 = eGraph.getOpInEClass<TransposeOp>(inputOp0);
    TransposeOp addInputOp1 = eGraph.getOpInEClass<TransposeOp>(inputOp1);
    if (!addInputOp0 || !addInputOp1) {
      return nullptr;
    }

    Value AddValue = rewriter
                         .create<AddOp>(op->getLoc(), addInputOp0.getOperand(),
                                        addInputOp1.getOperand())
                         .getResult();
    return rewriter.create<TransposeOp>(op->getLoc(), AddValue);
  }
};

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose, TransposeAddition, AdditionTranspose>(
      context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
