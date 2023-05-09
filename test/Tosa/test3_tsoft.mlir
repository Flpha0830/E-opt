module {
  func.func @test_tsoft() -> tensor<5x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with zeros
    %input1 = "tosa.const"() {value = dense<0.0> : tensor<5x3x3xf32>} : () -> tensor<5x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<5x3x3xf32>} : () -> tensor<5x3x3xf32>
    // Create a constant tensor for the permutation
    %perm = "tosa.const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>

    %t1 = "tosa.transpose"(%input1, %perm) : (tensor<5x3x3xf32>, tensor<3xi32>) -> tensor<3x3x5xf32>

    %t2 = "tosa.transpose"(%input2, %perm) : (tensor<5x3x3xf32>, tensor<3xi32>) -> tensor<3x3x5xf32>

    %soft = "tosa.add"(%t1, %t2) : (tensor<3x3x5xf32>, tensor<3x3x5xf32>) -> tensor<3x3x5xf32> 

    %tsoft = "tosa.transpose"(%soft, %perm) : (tensor<3x3x5xf32>, tensor<3xi32>) -> tensor<5x3x3xf32>

    // Return the result
    return %tsoft : tensor<5x3x3xf32>
  }
}
