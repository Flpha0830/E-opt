module {
  func.func @test_large() -> tensor<8x16x10xf32> {

    %a1 = "tosa.const"() {value = dense<1.0> : tensor<8x16x16xf32>} : () -> tensor<8x16x16xf32>
    %b1 = "tosa.const"() {value = dense<1.0> : tensor<8x16x32xf32>} : () -> tensor<8x16x32xf32>
    %c1 = "tosa.const"() {value = dense<1.0> : tensor<8x16x32xf32>} : () -> tensor<8x16x32xf32>
    %b2 = "tosa.const"() {value = dense<2.0> : tensor<8x32x64xf32>} : () -> tensor<8x32x64xf32>
    %c2 = "tosa.const"() {value = dense<2.0> : tensor<8x16x64xf32>} : () -> tensor<8x16x64xf32>
    %b3 = "tosa.const"() {value = dense<3.0> : tensor<8x64x10xf32>} : () -> tensor<8x64x10xf32>
    %c3 = "tosa.const"() {value = dense<3.0> : tensor<8x16x10xf32>} : () -> tensor<8x16x10xf32>

    %tmp0 = "tosa.matmul"(%a1, %b1) : (tensor<8x16x16xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %tmp1 = "tosa.add"(%tmp0, %c1) : (tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
    %tmp2 = "tosa.matmul"(%tmp1, %b2) : (tensor<8x16x32xf32>, tensor<8x32x64xf32>) -> tensor<8x16x64xf32>
    %tmp3 = "tosa.add"(%tmp2, %c2) : (tensor<8x16x64xf32>, tensor<8x16x64xf32>) -> tensor<8x16x64xf32>
    %tmp4 = "tosa.matmul"(%tmp3, %b3) : (tensor<8x16x64xf32>, tensor<8x64x10xf32>) -> tensor<8x16x10xf32>
    %tmp5 = "tosa.add"(%tmp4, %c3) : (tensor<8x16x10xf32>, tensor<8x16x10xf32>) -> tensor<8x16x10xf32>

    // Return the result
    return %tmp5 : tensor<8x16x10xf32>
  }
}
