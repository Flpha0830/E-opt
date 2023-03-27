//===- EGraph.cpp - EGraph Pattern classes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EGraph/EGraph.h"
#include "EGraph/Utils/OpIterator.h"

using namespace mlir;

void EGraph::buildWithOp(Operation *op) {
  OpIterator<TraversalOrder::PostOrder> it(op);
  for (; !it.isEnd(); ++it) {
    auto curr = *it;
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
  rootEClassId = eClassMap.size() - 1;
}

void EGraph::apply(
    DenseMap<OperationName,
             SmallVector<const OpEGraphRewritePattern<Operation> *, 2>>
        &patterns,
    PatternRewriter &rewriter, RewriterBase::Listener &listener) {

  std::vector<std::pair<Operation *, int64_t>> matches;

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

    for (auto eClassId : eNode->children) {
      for (auto eNode : eClassMap[eClassId]) {
        queue.push(eNode);
      }
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

void EGraph::rewriteWithBest(PatternRewriter &rewriter) {
  auto eNode = eClassMap[rootEClassId][0];
  std::set<ENode *> seen;
  seen.insert(eNode);

  int tmp = 0;
  int *depth = &tmp;
  ENode *ret = extractOp(eNode, depth, seen);

  std::stack<ENode *> stack;
  std::stack<unsigned> prevOperandIdx;
  ENode *curr = ret;
  std::queue<Operation *> localEraseOpList;

  while (curr != nullptr || !stack.empty()) {
    if (curr != nullptr) {
      stack.push(curr);
      prevOperandIdx.push(0);
      curr = curr->operand.size() > 0 ? curr->operand.at(0) : nullptr;
    } else {
      curr = stack.top();
      unsigned currOperandIndex = prevOperandIdx.top() + 1;
      if (currOperandIndex < curr->operand.size()) {
        prevOperandIdx.pop();
        prevOperandIdx.push(currOperandIndex);
        curr = curr->operand.at(currOperandIndex);
      } else {
        for (size_t i = 0; i < curr->operand.size(); i++) {
          auto it = eraseOpList.begin();
          for (; it != eraseOpList.end(); it++) {
            if ((*it) == curr->op->getOperand(i).getDefiningOp()) {
              break;
            }
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

        for (auto it = eraseOpList.begin(); it != eraseOpList.end(); it++) {
          if ((*it) == curr->op) {
            rewriter.setInsertionPoint(ret->op);

            curr->op = rewriter.create(
                curr->op->getLoc(), curr->op->getName().getIdentifier(),
                curr->op->getOperands(), curr->op->getResultTypes());
          }
        }

        stack.pop();
        prevOperandIdx.pop();
        curr = nullptr;
      }
    }
  }

  while (!localEraseOpList.empty()) {
    auto op = localEraseOpList.front();
    localEraseOpList.pop();

    for (Value operand : op->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        if (producer->hasOneUse())
          localEraseOpList.push(producer);
      }
    }
    op->erase();
  }
}

ENode *EGraph::extractOp(ENode *eNode, int *depth, std::set<ENode *> &seen) {
  if (eNode->children.size() == 0) {
    *depth = 1;
    return eNode;
  }

  *depth = INT_MIN;
  for (size_t i = 0; i < eNode->children.size(); i++) {
    auto eClassId = eNode->children[i];

    ENode *minChildENode = nullptr;
    int minVal = INT_MAX;
    for (auto eNode : eClassMap[eClassId]) {
      if (seen.count(eNode)) {
        continue;
      }
      seen.insert(eNode);

      int tmp = 0;
      int *val = &tmp;
      ENode *childOp = extractOp(eNode, val, seen);
      seen.erase(eNode);
      if (*val < minVal && childOp) {
        minVal = *val;
        minChildENode = childOp;
      }
    }

    if (minVal == INT_MAX || !minChildENode) {
      *depth = INT_MAX;
      return nullptr;
    }

    *depth = std::max(*depth, minVal);
    if (eNode->operand.size() <= i) {
      eNode->operand.push_back(minChildENode);
    }
  }

  *depth = *depth + 1;
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

  OpIterator<TraversalOrder::PostOrder> it(op);
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

  OpIterator<TraversalOrder::PostOrder> it(op);
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
