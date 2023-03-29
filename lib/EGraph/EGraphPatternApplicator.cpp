//===- EGraphPatternApplicator.cpp - Pattern Application Engine -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EGraph/EGraphPatternApplicator.h"
#include "EGraph/EGraph.h"
#include "EGraph/OpEGraphRewritePattern.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "egraph-pattern-application"

using namespace mlir;

EGraphPatternApplicator::EGraphPatternApplicator(
    const FrozenRewritePatternSet &frozenPatternList)
    : frozenPatternList(frozenPatternList) {}
EGraphPatternApplicator::~EGraphPatternApplicator() = default;

#ifndef NDEBUG
/// Log a message for a pattern that is impossible to match.
static void logImpossibleToMatch(const Pattern &pattern) {
  llvm::dbgs() << "Ignoring pattern '" << pattern.getRootKind()
               << "' because it is impossible to match or cannot lead "
                  "to legal IR (by cost model)\n";
}
#endif

LogicalResult
EGraphPatternApplicator::matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter,
                                         RewriterBase::Listener &listener) {
  // Copy over the patterns so that we can sort by benefit based on the cost
  // model. Patterns that are already impossible to match are ignored.
  patterns.clear();
  for (const auto &it : frozenPatternList.getOpSpecificNativePatterns()) {
    for (const auto &pattern : it.second) {
      if (const auto eGraphPattern =
              dyn_cast<OpEGraphRewritePattern<Operation>>(pattern)) {
        if (eGraphPattern->getBenefit().isImpossibleToMatch())
          LLVM_DEBUG(logImpossibleToMatch(*eGraphPattern));
        else
          patterns[it.first].push_back(eGraphPattern);
      }
    }
  }
  rewriter.setInsertionPoint(op);

  EGraph eGraph;
  eGraph.buildWithOp(op);
  size_t prevNumENode;
  size_t prevNumEClass;

  do {
    prevNumENode = eGraph.getNumENode();
    prevNumEClass = eGraph.getNumEClass();
    eGraph.apply(patterns, rewriter, listener);
  } while (eGraph.getNumENode() != prevNumENode ||
           eGraph.getNumEClass() != prevNumEClass);

  eGraph.rewriteWithBest(rewriter);
  return success();
}
