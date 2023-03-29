//===- OpIterator.h - OpIterator --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPITERATOR_H
#define OPITERATOR_H

#include "mlir/IR/Operation.h"

#include <queue>
#include <set>
#include <stack>

using namespace mlir;

enum class OpTraversalOrder { PreOrder, PostOrder };

template <OpTraversalOrder Order>
class OpIterator;

template <>
class OpIterator<OpTraversalOrder::PostOrder> {
public:
  OpIterator(Operation *op) {
    curr = op;
    traverseToLeftmostLeaf();
  }

  Operation *operator*() { return value; }

  OpIterator &operator++() {
    traverseToLeftmostLeaf();
    return *this;
  }

  bool isEnd() { return value == nullptr; }

private:
  std::stack<Operation *> stack;
  std::stack<unsigned> prevOperandIdx;
  Operation *curr;
  Operation *value;

  void traverseToLeftmostLeaf() {
    while (curr != nullptr || !stack.empty()) {
      if (curr != nullptr) {
        stack.push(curr);
        prevOperandIdx.push(0);
        curr = curr->getNumOperands() > 0 ? curr->getOperand(0).getDefiningOp()
                                          : nullptr;
      } else {
        curr = stack.top();
        unsigned currOperandIndex = prevOperandIdx.top() + 1;
        if (currOperandIndex < curr->getNumOperands()) {
          prevOperandIdx.pop();
          prevOperandIdx.push(currOperandIndex);
          curr = curr->getOperand(currOperandIndex).getDefiningOp();
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
class OpIterator<OpTraversalOrder::PreOrder> {
public:
  OpIterator(Operation *op) {
    value = op;
    traverseLeaves();
  }

  Operation *operator*() { return value; }

  OpIterator &operator++() {
    if (queue.empty()) {
      value = nullptr;
      return *this;
    }

    value = queue.front();
    queue.pop();
    traverseLeaves();
    return *this;
  }

  bool isEnd() { return value == nullptr; }

private:
  std::queue<Operation *> queue;
  Operation *value;

  void traverseLeaves() {
    for (Value operand : value->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        queue.push(producer);
      }
    }
  }
};

#endif // OPITERATOR_H
