// RUN: toy-opt %s | toy-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = toy.foo %{{.*}} : i32
        %res = toy.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @toy_types(%arg0: !toy.custom<"10">)
    func.func @toy_types(%arg0: !toy.custom<"10">) {
        return
    }
}
