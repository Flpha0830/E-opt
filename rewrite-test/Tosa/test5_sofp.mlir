module {
  func.func @test_pofs() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with ones
    %input1 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %input3 = "tosa.const"() {value = dense<2.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %input4 = "tosa.const"() {value = dense<3.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %p1 = "tosa.matmul"(%input1, %input2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %p2 = "tosa.matmul"(%input1, %input3) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>

    %sum = "tosa.add"(%p1, %p2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %sum2 = "tosa.add"(%sum, %input4) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>


    // Return the result
    return %sum2 : tensor<3x3x3xf32>
  }
}
