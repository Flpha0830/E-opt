module {
	func.func @test_random_2() ->tensor<2x64x16xf32> {
		%input2 =  "tosa.const"() {value = dense<2.0>: tensor<2x64x4xf32>} : () -> tensor<2x64x4xf32>
		%input7 =  "tosa.const"() {value = dense<8.0>: tensor<2x4x4xf32>} : () -> tensor<2x4x4xf32>
		%MatMul3 =  "tosa.matmul"(%input2, %input7) : (tensor<2x64x4xf32>, tensor<2x4x4xf32>) -> tensor<2x64x4xf32>
		%Add8 =  "tosa.add"(%MatMul3, %input2) : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Mul13 =  "tosa.mul"(%MatMul3, %Add8) {shift = 0 : i32} : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Add14 =  "tosa.add"(%Mul13, %Mul13) : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%input4 =  "tosa.const"() {value = dense<4.0>: tensor<2x4x16xf32>} : () -> tensor<2x4x16xf32>
		%MatMul17 =  "tosa.matmul"(%Add14, %input4) : (tensor<2x64x4xf32>, tensor<2x4x16xf32>) -> tensor<2x64x16xf32>
		return %MatMul17 : tensor<2x64x16xf32>
	}
}
