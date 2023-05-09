module {
  func.func @test_associativity_of_add() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with ones
    %input1 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %input3 = "tosa.const"() {value = dense<2.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>

    %sum1 = "tosa.add"(%input1, %input2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %sum2 = "tosa.add"(%sum1, %input3) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    // Return the result
    return %sum2 : tensor<3x3x3xf32>
  }
}
