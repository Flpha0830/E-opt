//===- ToyDialect.h - Toy dialect -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT_TOY_IR_TOYDIALECT_H
#define TOY_DIALECT_TOY_IR_TOYDIALECT_H

#include "Dialect/Toy/Transforms/ShapeInferenceInterface.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "Dialect/Toy/IR/ToyOpsDialect.h.inc"

#endif // TOY_DIALECT_TOY_IR_TOYDIALECT_H
