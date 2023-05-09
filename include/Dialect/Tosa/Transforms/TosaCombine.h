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
      if (perm0 == perm1) {
        llvm::dbgs() << "Rewrite ATT to A, with A = " << *ret.getDefiningOp()
                     << "\n";
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
      mlir::Value lt = rewriter
                           .create<TransposeOp>(op->getLoc(), result.getType(),
                                                AddInputOp.getOperand(0), perm)
                           .getResult();
      mlir::Value rt = rewriter
                           .create<TransposeOp>(op->getLoc(), result.getType(),
                                                AddInputOp.getOperand(1), perm)
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
    if (!addInputOp0 || !addInputOp1) {
      llvm::dbgs() << "Not a soft\n";
      return nullptr;
    }
    auto perm0 = addInputOp0.getOperand(1);
    auto perm1 = addInputOp1.getOperand(1);
    auto perm0Value = dyn_cast<ConstOp>(perm0.getDefiningOp()).getValue();
    auto perm1Value = dyn_cast<ConstOp>(perm1.getDefiningOp()).getValue();
    if ((perm0 != perm1) && (perm0Value != perm1Value)) {
      llvm::dbgs() << "Not a soft, different perm for T\n";
      return nullptr;
    }
    auto AddValue =
        rewriter
            .create<AddOp>(op->getLoc(), addInput0.getType(),
                           addInputOp0.getOperand(0), addInputOp1.getOperand(0))
            .getResult();
    llvm::dbgs() << "Rewrite a soft to tofs!\n";
    return rewriter.create<TransposeOp>(op->getLoc(), result.getType(),
                                        AddValue, perm0);
  }
};

struct ProdOfSumToSumOfProd : public mlir::OpEGraphRewritePattern<MatMulOp> {
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
    if (!SumOp) {
      llvm::dbgs() << "Not a pofs\n";
      return nullptr;
    }
    auto A = SumOp.getOperand(0);
    auto B = SumOp.getOperand(1);
    auto AC = rewriter.create<MatMulOp>(op->getLoc(), result.getType(), A, C)
                  .getResult();
    auto BC = rewriter.create<MatMulOp>(op->getLoc(), result.getType(), B, C)
                  .getResult();
    llvm::dbgs() << "Rewrite a pofs to sofp!\n";
    return rewriter.create<AddOp>(op->getLoc(), result.getType(), AC, BC);
  }
};

struct SumOfProdToProdOfSum : public mlir::OpEGraphRewritePattern<AddOp> {

  SumOfProdToProdOfSum(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a sofp\n";
    // A * B + C * D

    auto AB = op->getOperand(0);
    auto ABOp = AB.getDefiningOp();
    auto CD = op->getOperand(1);
    auto CDOp = CD.getDefiningOp();

    auto result = op->getResult(0);

    MatMulOp MulLeft = eGraph.getOpInEClass<MatMulOp>(ABOp);
    MatMulOp MulRight = eGraph.getOpInEClass<MatMulOp>(CDOp);

    if (!MulLeft || !MulRight) {
      llvm::dbgs() << "Not a sofp\n";
      return nullptr;
    }
    auto A = MulLeft.getOperand(0);
    auto B = MulLeft.getOperand(1);
    auto C = MulRight.getOperand(0);
    auto D = MulRight.getOperand(1);
    llvm::dbgs() << "A = " << *A.getDefiningOp() << "\n";
    llvm::dbgs() << "B = " << *B.getDefiningOp() << "\n";
    llvm::dbgs() << "C = " << *C.getDefiningOp() << "\n";
    llvm::dbgs() << "D = " << *D.getDefiningOp() << "\n";
    if (A == C) {
      llvm::dbgs() << "Find a sofp with A = C\n";
      auto BPlusD =
          rewriter.create<AddOp>(op->getLoc(), B.getType(), B, D).getResult();
      return rewriter.create<MatMulOp>(op->getLoc(), result.getType(), A,
                                       BPlusD);
    } else if (B == D) {
      llvm::dbgs() << "Find a sofp with B = D\n";
      auto APlusC =
          rewriter.create<AddOp>(op->getLoc(), C.getType(), A, C).getResult();
      return rewriter.create<MatMulOp>(op->getLoc(), result.getType(), APlusC,
                                       B);
    } else {
      llvm::dbgs() << "Not a sofp\n";
      return nullptr;
    }
  }
};

struct EWProdOfSumToSumOfEWProd : public mlir::OpEGraphRewritePattern<MulOp> {
  EWProdOfSumToSumOfEWProd(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<MulOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a ewpofs\n";
    auto L = op->getOperand(0);
    auto LOp = L.getDefiningOp();
    auto C = op->getOperand(1);
    auto shift = dyn_cast<IntegerAttr>(op->getAttr("shift"));
    auto result = op->getResult(0);
    AddOp SumOp = eGraph.getOpInEClass<AddOp>(LOp);
    if (!SumOp) {
      llvm::dbgs() << "Not a ewpofs\n";
      return nullptr;
    }
    // llvm::dbgs() << "shift : " << shift << "\n";
    auto A = SumOp.getOperand(0);
    auto B = SumOp.getOperand(1);
    auto AC =
        rewriter.create<MulOp>(op->getLoc(), result.getType(), A, C, shift)
            .getResult();
    auto BC =
        rewriter.create<MulOp>(op->getLoc(), result.getType(), B, C, shift)
            .getResult();
    llvm::dbgs() << "Rewrite a ewpofs to soewpf!\n";
    return rewriter.create<AddOp>(op->getLoc(), result.getType(), AC, BC);
    return nullptr;
  }
};

struct SumOfEWProdToEWProdOfSum : public mlir::OpEGraphRewritePattern<AddOp> {

  SumOfEWProdToEWProdOfSum(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    llvm::dbgs() << "Try to find a soewpf\n";
    // A * B + C * D

    auto AB = op->getOperand(0);
    auto ABOp = AB.getDefiningOp();
    auto CD = op->getOperand(1);
    auto CDOp = CD.getDefiningOp();

    auto result = op->getResult(0);

    MulOp MulLeft = eGraph.getOpInEClass<MulOp>(ABOp);
    MulOp MulRight = eGraph.getOpInEClass<MulOp>(CDOp);

    if (!MulLeft || !MulRight) {
      llvm::dbgs() << "Not a soewpf\n";
      return nullptr;
    }
    auto A = MulLeft.getOperand(0);
    auto B = MulLeft.getOperand(1);
    auto C = MulRight.getOperand(0);
    auto D = MulRight.getOperand(1);
    auto shift1 = dyn_cast<IntegerAttr>(MulLeft->getAttr("shift"));
    auto shift2 = dyn_cast<IntegerAttr>(MulRight->getAttr("shift"));
    if (shift1 != shift2) {
      llvm::dbgs() << "Not a soewpf, different shift\n";
      return nullptr;
    }
    // auto shift = dyn_cast<IntegerAttr>(MulLeft.getAttr("shift"));
    if (A == C) {
      llvm::dbgs() << "Find a soewpf with A = C\n";
      auto BPlusD =
          rewriter.create<AddOp>(op->getLoc(), B.getType(), B, D).getResult();
      return rewriter.create<MulOp>(op->getLoc(), result.getType(), A, BPlusD,
                                    shift1);
    } else if (B == D) {
      llvm::dbgs() << "Find a soewpf with B = D\n";
      auto APlusC =
          rewriter.create<AddOp>(op->getLoc(), C.getType(), A, C).getResult();
      return rewriter.create<MulOp>(op->getLoc(), result.getType(), APlusC, B,
                                    shift1);
    } else {
      llvm::dbgs() << "Not a soewpf\n";
      return nullptr;
    }
  }
};

struct CommutativityOfAdd : public mlir::OpEGraphRewritePattern<AddOp> {
  CommutativityOfAdd(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {
    auto L = op->getOperand(0);
    auto R = op->getOperand(1);
    auto result = op->getResult(0);

    llvm::dbgs() << "Find a commutativity of add\n";
    return rewriter.create<AddOp>(op->getLoc(), result.getType(), R, L);
  }
};

struct CommutativityOfEWProd : public mlir::OpEGraphRewritePattern<MulOp> {
  CommutativityOfEWProd(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<MulOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {

    auto L = op->getOperand(0);
    auto R = op->getOperand(1);
    auto shift = dyn_cast<IntegerAttr>(op->getAttr("shift"));
    auto result = op->getResult(0);

    llvm::dbgs() << "Find a commutativity of ewprod\n";
    return rewriter.create<MulOp>(op->getLoc(), result.getType(), R, L, shift);
  }
};

struct AssociativityOfAdd : public mlir::OpEGraphRewritePattern<AddOp> {
  AssociativityOfAdd(mlir::MLIRContext *context)
      : OpEGraphRewritePattern<AddOp>(context, 1) {}
  Operation *matchAndReturnSubst(Operation *op, PatternRewriter &rewriter,
                                 EGraph &eGraph) const override {

    auto L = op->getOperand(0);
    auto R = op->getOperand(1);
    auto result = op->getResult(0);

    llvm::dbgs() << "Try to find a associativity of add\n";
    if (AddOp AddOpL = eGraph.getOpInEClass<AddOp>(L.getDefiningOp())) {
      llvm::dbgs()
          << "Find a associativity of add, (A + B) + C => A + (B + C)\n";
      auto LL = AddOpL.getOperand(0);
      auto LR = AddOpL.getOperand(1);
      auto LRR = rewriter.create<AddOp>(op->getLoc(), LL.getType(), LR, R);
      llvm::dbgs() << "LRR = " << LRR << "\n";
      return rewriter.create<AddOp>(op->getLoc(), result.getType(), LL, LRR);
    } else if (AddOp AddOpR = eGraph.getOpInEClass<AddOp>(R.getDefiningOp())) {
      llvm::dbgs()
          << "Find a associativity of add, A + (B + C) => (A + B) + C\n";
      auto RL = AddOpR.getOperand(0);
      auto RR = AddOpR.getOperand(1);
      auto LRL = rewriter.create<AddOp>(op->getLoc(), RR.getType(), L, RL);
      llvm::dbgs() << "LRL = " << LRL << "\n";
      return rewriter.create<AddOp>(op->getLoc(), result.getType(), LRL, RR);
    } else {
      llvm::dbgs() << "Not a associativity of add: " << *op << "\n";
      return nullptr;
    }
  }
};

void getOpEGraphRewritePatterns(RewritePatternSet &results,
                                MLIRContext *context) {
  results.add<SimplifyRedundantTranspose, TransposeOfSumToSumOfTranspose,
              SumOfTransposeToTransposeOfSum, ProdOfSumToSumOfProd,
              SumOfProdToProdOfSum, EWProdOfSumToSumOfEWProd,
              SumOfEWProdToEWProdOfSum, CommutativityOfEWProd,
              CommutativityOfAdd, AssociativityOfAdd>(context);
}

void getOpCostMap(std::map<StringRef, int64_t> &map) {
  map.emplace("tosa.const", 0);
  map.emplace("func.return", 0);
}
