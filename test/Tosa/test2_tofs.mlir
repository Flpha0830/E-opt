module {
  func.func @test_soft() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with zeros
    %input1 = "tosa.const"() {value = dense<0.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant tensor for the permutation
    %perm1 = "tosa.const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>

    %sum = "tosa.add"(%input1, %input2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32> 

    %tofs = "tosa.transpose"(%sum, %perm1) : (tensor<3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3xf32>

    // Return the result
    return %tofs : tensor<3x3x3xf32>
  }
}
