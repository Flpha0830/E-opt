module {
	func.func @test_random_6() ->tensor<1x16x4xf32> {
		%input6 =  "tosa.const"() {value = dense<9.0>: tensor<1x16x4xf32>} : () -> tensor<1x16x4xf32>
		%input5 =  "tosa.const"() {value = dense<3.0>: tensor<1x4x4xf32>} : () -> tensor<1x4x4xf32>
		%MatMul2 =  "tosa.matmul"(%input6, %input5) : (tensor<1x16x4xf32>, tensor<1x4x4xf32>) -> tensor<1x16x4xf32>
		%Mul4 =  "tosa.mul"(%MatMul2, %input6) {shift = 0 : i32} : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		%Add11 =  "tosa.add"(%Mul4, %input6) : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		%Mul19 =  "tosa.mul"(%input6, %Add11) {shift = 0 : i32} : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		return %Mul19 : tensor<1x16x4xf32>
	}
}
