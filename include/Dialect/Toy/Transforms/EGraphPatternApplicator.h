//===- EGraphPatternApplicator.h - PatternApplicator ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EGRAPHPATTERNAPPLICATOR_H
#define EGRAPHPATTERNAPPLICATOR_H

#include "Dialect/Toy/Transforms/OpEGraphRewritePattern.h"

#include "mlir/Rewrite/FrozenRewritePatternSet.h"

namespace mlir {

class PatternRewriter;
class EGraphPatternApplicator {
public:
  explicit EGraphPatternApplicator(
      const FrozenRewritePatternSet &frozenPatternList);
  ~EGraphPatternApplicator();

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter,
                                RewriterBase::Listener &listener);

  /// Walk all of the patterns within the applicator.
  void walkAllPatterns(function_ref<void(const Pattern &)> walk);

private:
  /// The list that owns the patterns used within this applicator.
  const FrozenRewritePatternSet &frozenPatternList;
  /// The set of patterns to match for each operation, stable sorted by benefit.
  DenseMap<OperationName,
           SmallVector<const OpEGraphRewritePattern<Operation> *, 2>>
      patterns;
};
} // namespace mlir

#endif // EGRAPHPATTERNAPPLICATOR_H
