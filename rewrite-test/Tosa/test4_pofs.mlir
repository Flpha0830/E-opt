module {
  func.func @test_pofs() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with ones
    %input1 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %input3 = "tosa.const"() {value = dense<2.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant tensor for the permutation

    %sum = "tosa.add"(%input1, %input2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32> 

    %pofs = "tosa.matmul"(%sum, %input3) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>

    // Return the result
    return %pofs : tensor<3x3x3xf32>
  }
}
