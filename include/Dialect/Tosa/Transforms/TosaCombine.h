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
    mlir::Value result = op->getResult(0);
    Operation *inputOp = transposeInput.getDefiningOp();
    if (AddOp AddInputOp = eGraph.getOpInEClass<AddOp>(inputOp)) {
      auto perm = op->getOperand(1);
      mlir::Value lt =
          rewriter.create<TransposeOp>(op->getLoc(), result.getType() , AddInputOp.getOperand(0), perm)
              .getResult();
      mlir::Value rt =
          rewriter.create<TransposeOp>(op->getLoc(), result.getType() , AddInputOp.getOperand(1), perm)
              .getResult();
      llvm::dbgs() << "Rewrite a tofs to soft!\n";
      return rewriter.create<AddOp>(op->getLoc(), result.getType(), lt, rt);
    }
    llvm::dbgs() << "Not a tofs\n";       
    return nullptr;
  }
};


struct SumOfTransposeToTransposeOfSum
    : public mlir::OpEGraphRewritePattern<AddOp> {
  SumOfTransposeToTransposeOfSum(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a soft\n";
    auto addInput0 = op->getOperand(0);
    auto inputOp0 = addInput0.getDefiningOp();
    auto addInput1 = op->getOperand(1);
    auto inputOp1 = addInput1.getDefiningOp();
    auto result = op->getResult(0);
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
    auto AddValue = rewriter.create<AddOp>(op->getLoc(), addInput0.getType(), addInputOp0.getOperand(0), addInputOp1.getOperand(0)).getResult();
    llvm::dbgs() << "Rewrite a soft to tofs!\n";
    return rewriter.create<TransposeOp>(op->getLoc(), result.getType(), AddValue, perm0);
  }
};

struct ProdOfSumToSumOfProd
    : public mlir::OpEGraphRewritePattern<MatMulOp> {
  ProdOfSumToSumOfProd(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<MatMulOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a pofs\n";
    auto L = op->getOperand(0);
    auto LOp = L.getDefiningOp();
    auto C = op->getOperand(1);
    auto result = op->getResult(0);
    AddOp SumOp = eGraph.getOpInEClass<AddOp>(LOp);
    if(!SumOp){
      llvm::dbgs() << "Not a pofs\n";
      return nullptr;
    }
    auto A = SumOp.getOperand(0);
    auto B = SumOp.getOperand(1);
    auto AC = rewriter.create<MatMulOp>(op->getLoc(), result.getType(), A, C).getResult();
    auto BC = rewriter.create<MatMulOp>(op->getLoc(), result.getType(), B, C).getResult();
    llvm::dbgs() << "Rewrite a pofs to sofp!\n";
    return rewriter.create<AddOp>(op->getLoc(), result.getType(), AC, BC);
  }
};

void getOpEGraphRewritePatterns(RewritePatternSet &results,
                                MLIRContext *context) {
  results.add<SimplifyRedundantTranspose, TransposeOfSumToSumOfTranspose, SumOfTransposeToTransposeOfSum,
            ProdOfSumToSumOfProd>(context);
}

void getOpCostMap(std::map<StringRef, int64_t> &map) {
  map.emplace("tosa.const", 1);
}
