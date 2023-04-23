module {
  func.func @transpose_twice() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with zeros
    %input = "tosa.const"() {value = dense<0.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>

    // Create a constant tensor for the permutation
    %perm = "tosa.const"() {value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> tensor<3xi32>

    // First transpose operation
    %transpose1 = "tosa.transpose"(%input, %perm) : (tensor<3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3xf32>

    // Second transpose operation
    %transpose2 = "tosa.transpose"(%transpose1, %perm) : (tensor<3x3x3xf32>, tensor<3xi32>) -> tensor<3x3x3xf32>

    // Return the result
    return %transpose2 : tensor<3x3x3xf32>
  }
}
