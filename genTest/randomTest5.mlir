module {
	func.func @test_random_5() ->tensor<2x64x32xf32> {
		%input1 =  "tosa.const"() {value = dense<10.0>: tensor<2x64x8xf32>} : () -> tensor<2x64x8xf32>
		%Add0 =  "tosa.add"(%input1, %input1) : (tensor<2x64x8xf32>, tensor<2x64x8xf32>) -> tensor<2x64x8xf32>
		%Mul2 =  "tosa.mul"(%Add0, %input1) {shift = 0 : i32} : (tensor<2x64x8xf32>, tensor<2x64x8xf32>) -> tensor<2x64x8xf32>
		%Add3 =  "tosa.add"(%Add0, %Mul2) : (tensor<2x64x8xf32>, tensor<2x64x8xf32>) -> tensor<2x64x8xf32>
		%input9 =  "tosa.const"() {value = dense<2.0>: tensor<2x8x64xf32>} : () -> tensor<2x8x64xf32>
		%input3 =  "tosa.const"() {value = dense<6.0>: tensor<2x64x32xf32>} : () -> tensor<2x64x32xf32>
		%MatMul7 =  "tosa.matmul"(%input9, %input3) : (tensor<2x8x64xf32>, tensor<2x64x32xf32>) -> tensor<2x8x32xf32>
		%MatMul8 =  "tosa.matmul"(%Add3, %MatMul7) : (tensor<2x64x8xf32>, tensor<2x8x32xf32>) -> tensor<2x64x32xf32>
		return %MatMul8 : tensor<2x64x32xf32>
	}
}
