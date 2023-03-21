//===- Passes.h - Toy Passes Definition -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file exposes the entry points to create compiler passes for Toy.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_TOY_TRANSFORMS_PASSES_H
#define TOY_DIALECT_TOY_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;

std::unique_ptr<Pass> createEGraphCanonicalizerPass();

namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();
} // namespace toy
} // namespace mlir

#endif // TOY_DIALECT_TOY_TRANSFORMS_PASSES_H
