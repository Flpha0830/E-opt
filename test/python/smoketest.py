# RUN: %python %s | FileCheck %s

from mlir_toy.ir import *
from mlir_toy.dialects import (
  builtin as builtin_d,
  toy as toy_d
)

with Context():
  toy_d.register_dialect()
  module = Module.parse("""
    %0 = arith.constant 2 : i32
    %1 = toy.foo %0 : i32
    """)
  # CHECK: %[[C:.*]] = arith.constant 2 : i32
  # CHECK: toy.foo %[[C]] : i32
  print(str(module))
