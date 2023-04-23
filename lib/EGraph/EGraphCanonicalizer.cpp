//===- EGraphCanonicalizer.cpp - EGraph -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Toy/IR/ToyDialect.h"
#include "Dialect/Toy/IR/ToyOps.h"
#include "Dialect/Toy/IR/ToyTypes.h"
#include "Dialect/Toy/Transforms/Passes.h"
#include "EGraph/EGraphPatternApplicator.h"
#include "EGraph/OpEGraphRewritePattern.h"
#include "EGraph/Utils/OpIterator.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

// TODO
#include "Dialect/Tosa/Transforms/TosaCombine.h"

using namespace mlir;

namespace {
class EGraphPatternRewriteDriver : public PatternRewriter,
                                   public RewriterBase::Listener {
public:
  explicit EGraphPatternRewriteDriver(MLIRContext *ctx, Region *f,
                                      const FrozenRewritePatternSet &patterns);
  LogicalResult simplify() &&;

protected:
  /// Process ops until the worklist is empty or `config.maxNumRewrites` is
  /// reached. Return `true` if any IR was changed.
  bool processWorklist();

  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;

private:
  /// The low-level pattern applicator.
  EGraphPatternApplicator matcher;
  Region *f;
};

EGraphPatternRewriteDriver::EGraphPatternRewriteDriver(
    MLIRContext *ctx, Region *f, const FrozenRewritePatternSet &patterns)
    : PatternRewriter(ctx), matcher(patterns), f(f) {}

bool EGraphPatternRewriteDriver::processWorklist() {
  bool changed = false;

  while (!worklist.empty()) {
    auto *op = worklist.back();
    worklist.pop_back();
    if (succeeded(matcher.matchAndRewrite(op, *this, *this))) {
      changed = true;
    }
  }

  return changed;
}

LogicalResult EGraphPatternRewriteDriver::simplify() && {
  std::vector<Operation *> traverselist;
  DenseMap<Operation *, unsigned> traverselistMap;
  f->walk<>([&](mlir::Operation *op) { traverselist.push_back(op); });

  // Reverse the list so our pop-back loop processes them in-order.
  std::reverse(traverselist.begin(), traverselist.end());
  for (size_t i = 0, e = traverselist.size(); i != e; ++i) {
    if (traverselistMap.count(traverselist[i]))
      continue;
    Operation *rootOp = traverselist[i];
    worklist.push_back(rootOp);
    traverselistMap[rootOp] = i;

    OpIterator<OpTraversalOrder::PreOrder> it(rootOp);
    for (; !it.isEnd(); ++it) {
      auto curr = *it;
      traverselistMap[curr] = i;
    }
  }

  return success(processWorklist());
}

struct EGraphCanonicalizer
    : public mlir::PassWrapper<EGraphCanonicalizer, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EGraphCanonicalizer)

  StringRef getArgument() const final { return "e-graph"; }

  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // TODO: need a workaround for adding patterns of dialects in MLIR
    owningPatterns.add<SimplifyRedundantTranspose>(context);

    patterns = FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() override {
    Operation *op = getOperation();

    bool failed = false;
    for (Region &region : op->getRegions()) {
      EGraphPatternRewriteDriver driver(region.getContext(), &region, patterns);
      failed |= std::move(driver).simplify().failed();
    }

    if (failed)
      signalPassFailure();
  }

  FrozenRewritePatternSet patterns;
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::createEGraphCanonicalizerPass() {
  return std::make_unique<EGraphCanonicalizer>();
}
