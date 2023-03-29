//===- ENodeIterator.h - OpIterator -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENODEITERATOR_H
#define ENODEITERATOR_H

#include "EGraph/EGraph.h"

#include <queue>
#include <set>
#include <stack>

using namespace mlir;

enum class ENodeTraversalOrder { PreOrder, PostOrder };

template <ENodeTraversalOrder Order>
class ENodeIterator;

template <>
class ENodeIterator<ENodeTraversalOrder::PostOrder> {
public:
  ENodeIterator(ENode *eNode) {
    curr = eNode;
    traverseToLeftmostLeaf();
  }

  ENode *operator*() { return value; }

  ENodeIterator &operator++() {
    traverseToLeftmostLeaf();
    return *this;
  }

  bool isEnd() { return value == nullptr; }

private:
  std::stack<ENode *> stack;
  std::stack<unsigned> prevOperandIdx;
  ENode *curr;
  ENode *value;

  void traverseToLeftmostLeaf() {
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
          value = curr;
          stack.pop();
          prevOperandIdx.pop();
          curr = nullptr;
          return;
        }
      }
    }
    value = nullptr;
  }
};

template <>
class ENodeIterator<ENodeTraversalOrder::PreOrder> {
public:
  ENodeIterator(ENode *op,
                const std::map<int64_t, std::vector<ENode *>> &eClassMap) {
    value = op;
    this->eClassMap = eClassMap;
    traverseLeaves();
  }

  ENode *operator*() { return value; }

  ENodeIterator &operator++() {
    while (!queue.empty()) {
      auto eNode = queue.front();
      queue.pop();

      if (seen.count(eNode)) {
        continue;
      }
      seen.insert(eNode);

      value = eNode;
      traverseLeaves();
      return *this;
    }

    value = nullptr;
    return *this;
  }

  bool isEnd() { return value == nullptr; }

private:
  std::queue<ENode *> queue;
  std::set<ENode *> seen;
  ENode *value;
  std::map<int64_t, std::vector<ENode *>> eClassMap;

  void traverseLeaves() {
    for (auto eClassId : value->children) {
      for (auto eNode : eClassMap[eClassId]) {
        queue.push(eNode);
      }
    }
  }
};
#endif // ENODEITERATOR_H
