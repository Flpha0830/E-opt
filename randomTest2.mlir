module {
	func.func @test_random_2() ->tensor<2x4x128xf32> {
		%input0 =  "tosa.const"() {value = dense<5.0>: tensor<2x4x128xf32>} : () -> tensor<2x4x128xf32>
		%input2 =  "tosa.const"() {value = dense<9.0>: tensor<2x4x16xf32>} : () -> tensor<2x4x16xf32>
		%input4 =  "tosa.const"() {value = dense<1.0>: tensor<2x16x256xf32>} : () -> tensor<2x16x256xf32>
		%MatMul3 =  "tosa.matmul"(%input2, %input4) : (tensor<2x4x16xf32>, tensor<2x16x256xf32>) -> tensor<2x4x256xf32>
		%input1 =  "tosa.const"() {value = dense<9.0>: tensor<2x256x128xf32>} : () -> tensor<2x256x128xf32>
		%MatMul7 =  "tosa.matmul"(%MatMul3, %input1) : (tensor<2x4x256xf32>, tensor<2x256x128xf32>) -> tensor<2x4x128xf32>
		%Add8 =  "tosa.add"(%MatMul7, %MatMul7) : (tensor<2x4x128xf32>, tensor<2x4x128xf32>) -> tensor<2x4x128xf32>
		%Mul10 =  "tosa.mul"(%input0, %Add8) {shift = 0 : i32} : (tensor<2x4x128xf32>, tensor<2x4x128xf32>) -> tensor<2x4x128xf32>
		return %Mul10 : tensor<2x4x128xf32>
	}
}
