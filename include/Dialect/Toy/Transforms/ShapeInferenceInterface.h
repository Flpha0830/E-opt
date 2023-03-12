//===- ShapeInferenceInterface.h - Interface definitions for ShapeInference -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_TOY_TRANSFORMS_SHAPEINFERENCEINTERFACE_H_
#define TOY_DIALECT_TOY_TRANSFORMS_SHAPEINFERENCEINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace toy {

/// Include the auto-generated declarations.
#include "Dialect/Toy/Transforms/ShapeInferenceInterface.h.inc"

} // namespace toy
} // namespace mlir

#endif // TOY_DIALECT_TOY_TRANSFORMS_SHAPEINFERENCEINTERFACE_H_
