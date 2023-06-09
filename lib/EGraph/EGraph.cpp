//===- EGraph.cpp - EGraph Pattern classes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EGraph/EGraph.h"
#include "EGraph/OpEGraphRewritePattern.h"
#include "EGraph/Utils/ENodeIterator.h"
#include "EGraph/Utils/OpIterator.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

void EGraph::buildWithOp(Operation *op) {
  static std::set<ENode *> visited;
  OpIterator<OpTraversalOrder::PostOrder> it(op);
  for (; !it.isEnd(); ++it) {
    auto curr = *it;
    op2ENode.emplace(curr, curr);
    ENode *eNode = &op2ENode.at(curr);
    eNode->op = curr;

    for (Value operand : curr->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        auto child = &op2ENode.at(producer);
        int64_t childEClassId = eNode2EClass[child];
        if (visited.count(eNode)) {
          continue;
        }
        eNode->children.push_back(childEClassId);
      }
    }
    add(eNode);
    visited.insert(eNode);
    (eNode->op)->dump();
    llvm::dbgs() << "num children: " << eNode->children.size() << "\n";
  }
  rootEClassId = eClassMap.size() - 1;
}

void EGraph::apply(
    DenseMap<OperationName,
             SmallVector<const OpEGraphRewritePattern<Operation> *, 2>>
        &patterns,
    PatternRewriter &rewriter, RewriterBase::Listener &listener) {

  std::vector<std::pair<Operation *, int64_t>> matches;

  auto eNode = eClassMap[rootEClassId][0];
  ENodeIterator<ENodeTraversalOrder::PreOrder> it(eNode, eClassMap);
  for (; !it.isEnd(); ++it) {
    auto eNode = *it;
    auto op = eNode->op;
    SmallVector<const OpEGraphRewritePattern<Operation> *, 2> opPatterns;
    auto patternIt = patterns.find(op->getName());
    if (patternIt != patterns.end())
      opPatterns = patternIt->second;

    for (auto opPattern : opPatterns) {
      Operation *subst = opPattern->matchAndReturnSubst(op, rewriter, *this);
      if (!subst) {
        continue;
      }
      matches.push_back(std::make_pair(subst, eNode2EClass[eNode]));
    }
  }

  for (auto it = matches.begin(); it != matches.end(); it++) {
    int64_t eClassId1 = it->second;

    std::vector<ENode *>::iterator pIt;
    std::vector<ENode *> *parentEClass = &eClassMap[eClassId1];
    if (!op2ENode.count(it->first)) {
      for (pIt = parentEClass->begin(); pIt != parentEClass->end(); pIt++) {
        if (it->first->getName() == (*pIt)->op->getName() &&
            it->first->getNumOperands() == (*pIt)->children.size()) {
          break;
        }
      }
      if (pIt != parentEClass->end()) {
        erase(it->first);
        continue;
      }
    } else {
      auto eNode = &op2ENode.at(it->first);
      if (!eNode2EClass.count(eNode)) {
        erase(it->first);
        continue;
      }
    }

    int64_t eClassId2 = addSubst(it->first);
    merge(eClassId1, eClassId2);
  }

  rebuild();
}

void EGraph::rewriteWithBest(PatternRewriter &rewriter,
                             const std::map<StringRef, int64_t> &opCostMap) {
  auto eNode = eClassMap[rootEClassId][0];
  std::set<ENode *> seen;
  seen.insert(eNode);

  int tmp = 0;
  int *cost = &tmp;
  ENode *ret = extractOp(eNode, cost, seen, opCostMap);
  llvm::dbgs() << *(eNode->op) << " cost: " << *cost << "\n";
  std::queue<Operation *> localEraseOpList;
  ENodeIterator<ENodeTraversalOrder::PostOrder> it(ret);
  for (; !it.isEnd(); ++it) {
    auto curr = *it;
    Operation *latestDef = nullptr;
    for (size_t i = 0; i < curr->operand.size(); i++) {
      auto it = eraseOpList.begin();
      for (; it != eraseOpList.end();) {
        if ((*it) == curr->op->getOperand(i).getDefiningOp()) {
          break;
        }
        it = (*it) == curr->op ? eraseOpList.erase(it) : it + 1;
      }

      Operation *def = curr->operand[i]->op;
      if (def && (!latestDef || latestDef->isBeforeInBlock(def))) {
        latestDef = def;
      }

      // avoid reset same op
      if (curr->op->getOperand(i).getDefiningOp() == curr->operand[i]->op) {
        continue;
      }

      if (it == eraseOpList.end()) {
        localEraseOpList.push(curr->op->getOperand(i).getDefiningOp());
      }

      curr->op->setOperand(i, curr->operand[i]->op->getResult(0));
    }
    // Move 'op' after the latest operand definition, if there is one.
    if (latestDef) {
      curr->op->moveAfter(latestDef);
    }
  }

  while (!localEraseOpList.empty()) {
    auto op = localEraseOpList.front();
    localEraseOpList.pop();

    for (Value operand : op->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        // TODO: need more check
        auto user = producer->getUsers().begin();
        for (; user != producer->getUsers().end(); user++) {
          if (*user == op) {
            continue;
          }

          auto it = eraseOpList.begin();
          for (; it != eraseOpList.end(); it++) {
            if ((*user) == (*it)) {
              break;
            }
          }
          if (it != eraseOpList.end()) {
            continue;
          }
          break;
        }

        if (user == producer->getUsers().end()) {
          localEraseOpList.push(producer);
        }
      }
    }
    op->dropAllUses();
    op->erase();
  }
}

ENode *EGraph::extractOp(ENode *eNode, int *cost, std::set<ENode *> &seen,
                         const std::map<StringRef, int64_t> &opCostMap) {
  static std::map<ENode *, int> ENodeCostMap;
  if (eNode->children.size() == 0) {
    auto opName = eNode->op->getName().getStringRef();
    *cost = opCostMap.count(opName) ? opCostMap.at(opName) : 0;
    return eNode;
  }

  // *cost = 0;
  llvm::dbgs() << "Current Node:" << *(eNode->op)
               << " with number of children:" << (eNode->children).size()
               << "\n";
  for (size_t i = 0; i < eNode->children.size(); i++) {
    auto eClassId = eNode->children[i];

    ENode *minChildENode = nullptr;
    int minVal = INT_MAX;
    for (auto eNode : eClassMap[eClassId]) {
      if (seen.count(eNode)) {
        minVal = 0;
        minChildENode = eNode;
        continue;
      }
      seen.insert(eNode);

      int tmp = 0;
      int *val = &tmp;
      ENode *childOp = extractOp(eNode, val, seen, opCostMap);
      seen.erase(eNode);
      if (*val < minVal && childOp) {
        minVal = *val;
        minChildENode = childOp;
      }
    }

    if (minVal == INT_MAX || !minChildENode) {
      *cost = INT_MAX;
      return nullptr;
    }
    if (ENodeCostMap.count(minChildENode)) {
      minVal = 0;
    } else {
      ENodeCostMap[minChildENode] = minVal;
      *cost += minVal;
    }

    llvm::dbgs() << *(minChildENode->op) << " cost: " << minVal << "\n";

    if (eNode->operand.size() <= i) {
      llvm::dbgs() << "Choose operand:" << *(minChildENode->op)
                   << "with cost: " << minVal << "\n";
      eNode->operand.push_back(minChildENode);
    }
  }

  auto opName = eNode->op->getName().getStringRef();
  int64_t subCost = opCostMap.count(opName) ? opCostMap.at(opName) : 1;

  if (auto matmulop = dyn_cast<tosa::MatMulOp>(eNode->op)) {
    auto t1 = matmulop->getOperand(0).getType().cast<mlir::TensorType>();
    auto t2 = matmulop->getOperand(1).getType().cast<mlir::TensorType>();
    auto n = t1.getShape()[0];
    auto h = t1.getShape()[1];
    auto c = t1.getShape()[2];
    auto w = t2.getShape()[2];
    subCost = n * h * c * w * 3;
  } else if (auto addop = dyn_cast<tosa::AddOp>(eNode->op)) {
    auto t2 = addop->getOperand(1).getType().cast<mlir::TensorType>();
    auto n = t2.getShape()[0];
    auto h = t2.getShape()[1];
    auto w = t2.getShape()[2];
    subCost = n * h * w;
  } else if (auto mulop = dyn_cast<tosa::MulOp>(eNode->op)) {
    auto t2 = mulop->getOperand(1).getType().cast<mlir::TensorType>();
    auto n = t2.getShape()[0];
    auto h = t2.getShape()[1];
    auto w = t2.getShape()[2];
    subCost = n * h * w * 2;
  }
  // *cost = 0;
  // for(auto [k,v] : ENodeCostMap){
  //   *cost += v;
  // }
  *cost += subCost;

  llvm::dbgs() << "End current node:" << *(eNode->op) << "with cost: " << *cost
               << "\n and nops:" << eNode->operand.size() << "\n";

  return eNode;
}

void EGraph::dump() {
  auto eNode = eClassMap[rootEClassId][0];

  std::queue<ENode *> queue;
  std::set<ENode *> seen;
  queue.push(eNode);

  while (!queue.empty()) {
    auto eNode = queue.front();

    queue.pop();
    if (seen.count(eNode)) {
      continue;
    }
    seen.insert(eNode);

    int64_t eClassId = eNode2EClass[eNode];
    llvm::outs() << "ENode Name: " << eNode->op->getName() << "\n"
                 << "ENode EClass: " << eClassId << "\n"
                 << "ENode Children: ";
    for (size_t i = 0; i < eNode->children.size(); i++) {
      llvm::outs() << eNode->children[i] << " ";
      int64_t childEClassId = eNode->children[i];
      if (!eClassMap.count(childEClassId)) {
        continue;
      }
      for (auto const &it : eClassMap[childEClassId]) {
        queue.push(it);
      }
    }
    llvm::outs() << "\n";
  }

  llvm::outs() << "enode size: " << eNode2EClass.size() << "\n";
  llvm::outs() << "eclass size: " << eClassMap.size() << "\n";
  llvm::outs() << "rebuild size: " << rebuildList.size() << "\n";
  llvm::outs() << "erase size: " << eraseOpList.size() << "\n";
}

void EGraph::rebuild() {
  while (!rebuildList.empty()) {
    std::vector<int64_t> localRebuildList = rebuildList;
    rebuildList.clear();

    std::set<int64_t> seen;
    for (auto eClassId : localRebuildList) {
      if (seen.count(eClassId)) {
        continue;
      }
      seen.insert(eClassId);

      //            llvm::outs() << eClassId << "\n";
      int64_t newEClassId = find(eClassId);
      std::vector<ENode *> *eClass = &eClassMap[newEClassId];
      auto cmp = [](const ENode *a, const ENode *b) {
        if (a->op->getName() == b->op->getName() &&
            a->children.size() == b->children.size()) {
          return a == b;
        }
        return a->op < b->op;
      };
      std::set<ENode *, decltype(cmp)> s(cmp);
      for (auto it = eClass->begin(); it != eClass->end(); it++) {
        if (s.count(*it)) {
          eNode2EClass.erase(*it);
          continue;
        }
        s.insert(*it);
      }
      std::vector<ENode *> newEClass(s.begin(), s.end());
      eClassMap[newEClassId] = newEClass;

      for (auto &[parentENode, parentEClass] : eClassParents[eClassId]) {
        if (!eNode2EClass.count(parentENode)) {
          continue;
        }
        canonicalize(parentENode);
        eNode2EClass[parentENode] = find(parentEClass);
      }

      std::map<ENode *, int64_t> seen;
      for (auto &[parentENode, parentEClass] : eClassParents[eClassId]) {
        canonicalize(parentENode);
        if (seen.count(parentENode)) {
          merge(parentEClass, seen[parentENode]);
        }
        seen.emplace(parentENode, find(parentEClass));
      }
      std::vector<std::pair<ENode *, int64_t>> newParent(seen.begin(),
                                                         seen.end());
      eClassParents[eClassId] = newParent;
    }
  }
}

void EGraph::erase(Operation *op) {
  std::vector<Operation *> localEraseOpList;

  OpIterator<OpTraversalOrder::PostOrder> it(op);
  for (; !it.isEnd(); ++it) {
    auto curr = *it;
    if (!op2ENode.count(curr)) {
      localEraseOpList.push_back(curr);
    }
  }

  std::reverse(localEraseOpList.begin(), localEraseOpList.end());
  eraseOpList.insert(eraseOpList.end(), localEraseOpList.begin(),
                     localEraseOpList.end());
}

int64_t EGraph::addSubst(Operation *op) {
  std::vector<Operation *> localEraseOpList;

  OpIterator<OpTraversalOrder::PostOrder> it(op);
  for (; !it.isEnd(); ++it) {
    auto curr = *it;
    if (!op2ENode.count(curr)) {
      localEraseOpList.push_back(curr);

      op2ENode.emplace(curr, curr);
      ENode *eNode = &op2ENode.at(curr);
      eNode->op = curr;

      for (Value operand : curr->getOperands()) {
        if (Operation *producer = operand.getDefiningOp()) {
          auto child = &op2ENode.at(producer);
          int64_t childEClassId = eNode2EClass[child];
          eNode->children.push_back(childEClassId);
        }
      }
      add(eNode);
    }
  }

  std::reverse(localEraseOpList.begin(), localEraseOpList.end());
  eraseOpList.insert(eraseOpList.end(), localEraseOpList.begin(),
                     localEraseOpList.end());

  auto eNode = &op2ENode.at(op);
  int64_t eClassId = eNode2EClass[eNode];
  return eClassId;
}

void EGraph::canonicalize(ENode *eNode) {
  for (size_t i = 0; i < eNode->children.size(); i++) {
    eNode->children[i] = find(eNode->children[i]);
  }
}

int64_t EGraph::add(ENode *eNode) {
  canonicalize(eNode);
  if (eNode2EClass.count(eNode)) {
    return eNode2EClass[eNode];
  }

  int64_t eClassId = getNewEClassId();
  // update e-class
  eNode2EClass[eNode] = eClassId;

  // update e-class group
  eClassMap[eClassId].emplace_back(eNode);

  // update e-class parents
  for (int64_t child : eNode->children) {
    eClassParents[child].emplace_back(eNode, eClassId);
  }
  return eClassId;
}

void EGraph::merge(int64_t eClassId1, int64_t eClassId2) {
  if (find(eClassId1) == find(eClassId2)) {
    return;
  }
  int64_t newEClassId = unionFind.merge(eClassId1, eClassId2);

  std::vector<ENode *> *parentEClass = &eClassMap[eClassId1];
  std::vector<ENode *> *eClass = &eClassMap[eClassId2];
  for (auto it = eClass->begin(); it != eClass->end(); it++) {
    eNode2EClass[*it] = eClassId1;
    parentEClass->push_back(*it);
  }
  eClassMap.erase(eClassId2);

  rebuildList.push_back(newEClassId);
}

int64_t EGraph::find(int64_t eClassId) { return unionFind.find(eClassId); }

int64_t EGraph::getNewEClassId() {
  int64_t newEClassId = eClassId++;
  unionFind.add(newEClassId);
  return newEClassId;
}
