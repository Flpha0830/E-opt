module {
  func.func @test_associativity_of_matmul() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with ones
    %input1 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %input3 = "tosa.const"() {value = dense<2.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>

    %mm1 = "tosa.matmul"(%input1, %input2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %mm2 = "tosa.matmul"(%mm1, %input3) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    // Return the result
    return %mm2 : tensor<3x3x3xf32>
  }
}
