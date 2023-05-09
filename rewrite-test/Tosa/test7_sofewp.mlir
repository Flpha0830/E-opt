module {
  func.func @test_sofwep() -> tensor<3x3x3xf32> {
    // Create a constant 3x3x3 input tensor filled with ones
    %input1 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    // Create a constant 3x3x3 input tensor filled with ones
    %input2 = "tosa.const"() {value = dense<1.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>
    %input3 = "tosa.const"() {value = dense<2.0> : tensor<3x3x3xf32>} : () -> tensor<3x3x3xf32>

    %p1 = "tosa.mul"(%input1, %input3) {shift = 0 : i32} : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %p2 = "tosa.mul"(%input1, %input2) {shift = 0 : i32} : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32>
    %sum = "tosa.add"(%p1, %p2) : (tensor<3x3x3xf32>, tensor<3x3x3xf32>) -> tensor<3x3x3xf32> 
    // Return the result
    return %sum : tensor<3x3x3xf32>
  }
}
