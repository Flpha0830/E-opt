//===- TosaCombine.h - Toy High Level Optimizer ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Tosa dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"

#include "EGraph/EGraph.h"
#include "EGraph/OpEGraphRewritePattern.h"

using namespace mlir;
using namespace tosa;

struct SimplifyRedundantTranspose
    : public mlir::OpEGraphRewritePattern<TransposeOp> {
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    mlir::Value transposeInput = op->getOperand(0);
    Operation *inputOp = transposeInput.getDefiningOp();

    if (TransposeOp transposeInputOp =
            eGraph.getOpInEClass<TransposeOp>(inputOp)) {
      mlir::Value ret = transposeInputOp.getOperand(0);
      return ret.getDefiningOp();
    }
    return nullptr;
  }
};

void getOpEGraphRewritePatterns(RewritePatternSet &results,
                                MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

void getOpCostMap(std::map<StringRef, int64_t> &map) {
  map.emplace("tosa.const", 1);
}