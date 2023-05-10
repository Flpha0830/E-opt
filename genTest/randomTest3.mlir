module {
	func.func @test_random_3() ->tensor<4x32x256xf32> {
		%input1 =  "tosa.const"() {value = dense<6.0>: tensor<4x32x128xf32>} : () -> tensor<4x32x128xf32>
		%input3 =  "tosa.const"() {value = dense<3.0>: tensor<4x128x16xf32>} : () -> tensor<4x128x16xf32>
		%MatMul1 =  "tosa.matmul"(%input1, %input3) : (tensor<4x32x128xf32>, tensor<4x128x16xf32>) -> tensor<4x32x16xf32>
		%input4 =  "tosa.const"() {value = dense<9.0>: tensor<4x32x16xf32>} : () -> tensor<4x32x16xf32>
		%Add9 =  "tosa.add"(%MatMul1, %input4) : (tensor<4x32x16xf32>, tensor<4x32x16xf32>) -> tensor<4x32x16xf32>
		%Mul16 =  "tosa.mul"(%Add9, %input4) {shift = 0 : i32} : (tensor<4x32x16xf32>, tensor<4x32x16xf32>) -> tensor<4x32x16xf32>
		%Add21 =  "tosa.add"(%Mul16, %Add9) : (tensor<4x32x16xf32>, tensor<4x32x16xf32>) -> tensor<4x32x16xf32>
		%input7 =  "tosa.const"() {value = dense<5.0>: tensor<4x16x256xf32>} : () -> tensor<4x16x256xf32>
		%MatMul24 =  "tosa.matmul"(%Add21, %input7) : (tensor<4x32x16xf32>, tensor<4x16x256xf32>) -> tensor<4x32x256xf32>
		return %MatMul24 : tensor<4x32x256xf32>
	}
}
