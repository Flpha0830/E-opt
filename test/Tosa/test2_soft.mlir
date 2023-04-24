module {
  func.func @test_soft() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with zeros
    %input1 = "tosa.const"() {value = dense<0.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant tensor for the permutation
    %perm1 = "tosa.const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>

    %t1 = "tosa.transpose"(%input1, %perm1) : (tensor<3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3xf32>

    %t2 = "tosa.transpose"(%input2, %perm1) : (tensor<3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3xf32>

    %soft = "tosa.add"(%t1, %t2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32> 

    // Return the result
    return %soft : tensor<3x3x3xf32>
  }
}
