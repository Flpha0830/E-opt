//===- ToyTypes.h - Toy dialect types ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_TOY_IR_TOYTYPES_H
#define TOY_DIALECT_TOY_IR_TOYTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Toy/IR/ToyOpsTypes.h.inc"

#endif // TOY_DIALECT_TOY_IR_TOYTYPES_H
