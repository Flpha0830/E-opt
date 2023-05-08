//===- EGraph.h - EGraph Pattern classes ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EGRAPH_H
#define EGRAPH_H

#include "EGraph/OpEGraphRewritePattern.h"

#include "mlir/IR/PatternMatch.h"
#include <queue>
#include <set>
#include <stack>

namespace mlir {
struct ENode {
  std::vector<int64_t> children;
  Operation *op;
  std::vector<ENode *> operand;

  ENode(Operation *op) : op(op) {}

  bool operator<(const ENode &rhs) const { return this->op < rhs.op; }
};

class EGraph {
private:
  class UnionFind {
  public:
    void add(int64_t id) { eClassParent[id] = id; }
    int64_t find(int64_t id) {
      return id == eClassParent[id]
                 ? id
                 : (eClassParent[id] = find(eClassParent[id]));
    }
    int64_t merge(int64_t id1, int64_t id2) {
      int64_t rootId1 = find(id1);
      int64_t rootId2 = find(id2);
      if (rootId1 != rootId2) {
        eClassParent[rootId2] = rootId1;
      }
      return rootId2;
    }

  private:
    std::map<int64_t, int64_t> eClassParent;
  };

public:
  explicit EGraph() {}
  ~EGraph() {
    for (auto it = eraseOpList.begin(); it != eraseOpList.end(); it++) {
      (*it)->dropAllUses();
      (*it)->erase();
    }
  }

  void buildWithOp(Operation *op);
  void apply(DenseMap<OperationName,
                      SmallVector<const OpEGraphRewritePattern<Operation> *, 2>>
                 &patterns,
             PatternRewriter &rewriter, RewriterBase::Listener &listener);
  void rewriteWithBest(PatternRewriter &rewriter,
                       const std::map<StringRef, int64_t> &opCostMap);
  void dump();

  size_t getNumENode() { return eNode2EClass.size(); }
  size_t getNumEClass() { return eClassMap.size(); }

  template <typename OpTy>
  OpTy getOpInEClass(Operation *op) {
    auto eNode = &op2ENode.at(op);
    if (!eNode2EClass.count(eNode)) {
      return nullptr;
    }
    int64_t eClassId = eNode2EClass[eNode];
    std::vector<ENode *> eNodesInEClass = eClassMap[eClassId];

    for (auto it = eNodesInEClass.begin(); it != eNodesInEClass.end(); it++) {
      if (OpTy opInEClass = dyn_cast<OpTy>((*it)->op)) {
        return opInEClass;
      }
    }
    return nullptr;
  }

private:
  int64_t eClassId = 0;
  int64_t rootEClassId = 0;
  std::map<Operation *, ENode> op2ENode;
  std::map<ENode *, int64_t> eNode2EClass;
  std::map<int64_t, std::vector<ENode *>> eClassMap;
  std::map<int64_t, std::vector<std::pair<ENode *, int64_t>>> eClassParents;
  std::vector<Operation *> eraseOpList;
  std::vector<int64_t> rebuildList;
  UnionFind unionFind;

  void rebuild();
  ENode *extractOp(ENode *eNode, int *depth, std::set<ENode *> &seen,
                   const std::map<StringRef, int64_t> &opCostMap);
  void erase(Operation *op);
  int64_t addSubst(Operation *op);

  void canonicalize(ENode *eNode);
  int64_t add(ENode *eNode);
  void merge(int64_t eClassId1, int64_t eClassId2);
  int64_t find(int64_t eClassId);

  int64_t getNewEClassId();
};
} // namespace mlir

#endif // EGRAPH_H
