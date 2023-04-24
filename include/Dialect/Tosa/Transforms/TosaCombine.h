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
      : OpEGraphRewritePattern<TransposeOp>(context, /*benefit=*/10) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a TofT\n";                                
    mlir::Value transposeInput = op->getOperand(0);
    mlir::Value perm0 = op->getOperand(1);
    Operation *inputOp = transposeInput.getDefiningOp();

    if (TransposeOp transposeInputOp =
            eGraph.getOpInEClass<TransposeOp>(inputOp)) {
      mlir::Value ret = transposeInputOp.getOperand(0);
      mlir::Value perm1 = op->getOperand(1);
      if(perm0 == perm1){
        llvm::dbgs() << "Rewrite ATT to A, with A = " << *ret.getDefiningOp() <<"\n";
        return ret.getDefiningOp();
      }
      
    }
    llvm::dbgs() << "Not a TofT\n";  
    return nullptr;
  }
};

struct TransposeOfSumToSumOfTranspose
    : public mlir::OpEGraphRewritePattern<TransposeOp> {
  TransposeOfSumToSumOfTranspose(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<TransposeOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a tofs\n";                              
    mlir::Value transposeInput = op->getOperand(0);
    Operation *inputOp = transposeInput.getDefiningOp();
    if (AddOp AddInputOp = eGraph.getOpInEClass<AddOp>(inputOp)) {
      auto perm = op->getOperand(1);
      mlir::Value lt =
          rewriter.create<TransposeOp>(op->getLoc(), op->getResultTypes() , AddInputOp.getOperand(0), perm)
              .getResult();
      mlir::Value rt =
          rewriter.create<TransposeOp>(op->getLoc(), op->getResultTypes() , AddInputOp.getOperand(1), perm)
              .getResult();
      llvm::dbgs() << "Rewrite a tofs to soft!\n";
      return rewriter.create<AddOp>(op->getLoc(),op->getResultTypes(), lt, rt);
    }
    llvm::dbgs() << "Not a tofs\n";       
    return nullptr;
  }
};


struct SumOfTransposeToTransposeOfSum
    : public mlir::OpEGraphRewritePattern<AddOp> {
  SumOfTransposeToTransposeOfSum(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, 3) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a soft\n";
    auto addInput0 = op->getOperand(0);
    auto inputOp0 = addInput0.getDefiningOp();
    auto addInput1 = op->getOperand(1);
    auto inputOp1 = addInput1.getDefiningOp();
    auto addInputOp0 = eGraph.getOpInEClass<TransposeOp>(inputOp0);
    auto addInputOp1 = eGraph.getOpInEClass<TransposeOp>(inputOp1);
    if(!addInputOp0 || !addInputOp1){
      llvm::dbgs() << "Not a soft\n";
      return nullptr;
    }
    auto perm0 = addInputOp0.getOperand(1);
    auto perm1 = addInputOp1.getOperand(1);
    if(perm0 != perm1 && perm0){
      return nullptr;
    }
    auto AddValue = rewriter.create<AddOp>(op->getLoc(), op->getResultTypes(), addInputOp0.getOperand(0), addInputOp1.getOperand(0)).getResult();
    llvm::dbgs() << "Rewrite a soft to tofs!\n";
    return rewriter.create<TransposeOp>(op->getLoc(), op->getResultTypes(), AddValue, perm0);
  }
};

void getOpEGraphRewritePatterns(RewritePatternSet &results,
                                MLIRContext *context) {
  results.add<SimplifyRedundantTranspose, TransposeOfSumToSumOfTranspose, SumOfTransposeToTransposeOfSum>(context);
}

void getOpCostMap(std::map<StringRef, int64_t> &map) {
  map.emplace("tosa.const", 1);
}
