//===- OpEGraphRewritePattern.h - EGraph Pattern classes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPEGRAPHREWRITEPATTERN_H
#define OPEGRAPHREWRITEPATTERN_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class EGraph;

template <typename SourceOp>
struct OpEGraphRewritePattern : public OpRewritePattern<SourceOp> {
  OpEGraphRewritePattern(MLIRContext *context, PatternBenefit benefit = 1,
                         ArrayRef<StringRef> generatedNames = {"e-graph"})
      : OpRewritePattern<SourceOp>(context, benefit, generatedNames) {}
  using mlir::OpRewritePattern<SourceOp>::matchAndRewrite;

  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        PatternRewriter &rewriter) const final {
    return failure();
  }

  static bool classof(const Pattern *p) {
    ArrayRef<OperationName> ops = p->getGeneratedOps();
    if (ops.size() >= 1u && ops.front().getStringRef().equals("e-graph")) {
      return true;
    }
    return false;
  }

  virtual Operation *matchAndReturnSubst(Operation *op,
                                         PatternRewriter &rewriter,
                                         EGraph &eGraph) const {
    llvm_unreachable("must override rewrite or matchAndRewrite");
  }
};
} // namespace mlir

#endif // OPEGRAPHREWRITEPATTERN_H
