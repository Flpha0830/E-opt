module {
	func.func @test_random_1() ->tensor<4x16x256xf32> {
		%input3 =  "tosa.const"() {value = dense<6.0>: tensor<4x16x4xf32>} : () -> tensor<4x16x4xf32>
		%input4 =  "tosa.const"() {value = dense<4.0>: tensor<4x4x256xf32>} : () -> tensor<4x4x256xf32>
		%MatMul0 =  "tosa.matmul"(%input3, %input4) : (tensor<4x16x4xf32>, tensor<4x4x256xf32>) -> tensor<4x16x256xf32>
		%input7 =  "tosa.const"() {value = dense<3.0>: tensor<4x16x256xf32>} : () -> tensor<4x16x256xf32>
		%Mul1 =  "tosa.mul"(%MatMul0, %input7) {shift = 0 : i32} : (tensor<4x16x256xf32>, tensor<4x16x256xf32>) -> tensor<4x16x256xf32>
		%input6 =  "tosa.const"() {value = dense<8.0>: tensor<4x256x16xf32>} : () -> tensor<4x256x16xf32>
		%MatMul9 =  "tosa.matmul"(%Mul1, %input6) : (tensor<4x16x256xf32>, tensor<4x256x16xf32>) -> tensor<4x16x16xf32>
		%Mul3 =  "tosa.mul"(%Mul1, %MatMul0) {shift = 0 : i32} : (tensor<4x16x256xf32>, tensor<4x16x256xf32>) -> tensor<4x16x256xf32>
		%Add8 =  "tosa.add"(%input7, %Mul3) : (tensor<4x16x256xf32>, tensor<4x16x256xf32>) -> tensor<4x16x256xf32>
		%MatMul14 =  "tosa.matmul"(%MatMul9, %Add8) : (tensor<4x16x16xf32>, tensor<4x16x256xf32>) -> tensor<4x16x256xf32>
		%Mul20 =  "tosa.mul"(%MatMul14, %MatMul0) {shift = 0 : i32} : (tensor<4x16x256xf32>, tensor<4x16x256xf32>) -> tensor<4x16x256xf32>
		return %Mul20 : tensor<4x16x256xf32>
	}
}
