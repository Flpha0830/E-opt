module {
	func.func @test_random_3() ->tensor<4x128x8xf32> {
		%input2 =  "tosa.const"() {value = dense<3.0>: tensor<4x128x128xf32>} : () -> tensor<4x128x128xf32>
		%Add4 =  "tosa.add"(%input2, %input2) : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
		%Mul5 =  "tosa.mul"(%input2, %Add4) {shift = 0 : i32} : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
		%Mul9 =  "tosa.mul"(%input2, %Mul5) {shift = 0 : i32} : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
		%Mul10 =  "tosa.mul"(%Mul9, %input2) {shift = 0 : i32} : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
		%input7 =  "tosa.const"() {value = dense<1.0>: tensor<4x128x16xf32>} : () -> tensor<4x128x16xf32>
		%input8 =  "tosa.const"() {value = dense<7.0>: tensor<4x16x8xf32>} : () -> tensor<4x16x8xf32>
		%MatMul2 =  "tosa.matmul"(%input7, %input8) : (tensor<4x128x16xf32>, tensor<4x16x8xf32>) -> tensor<4x128x8xf32>
		%Mul12 =  "tosa.mul"(%MatMul2, %MatMul2) {shift = 0 : i32} : (tensor<4x128x8xf32>, tensor<4x128x8xf32>) -> tensor<4x128x8xf32>
		%MatMul14 =  "tosa.matmul"(%Mul10, %Mul12) : (tensor<4x128x128xf32>, tensor<4x128x8xf32>) -> tensor<4x128x8xf32>
		%MatMul17 =  "tosa.matmul"(%Mul9, %MatMul14) : (tensor<4x128x128xf32>, tensor<4x128x8xf32>) -> tensor<4x128x8xf32>
		return %MatMul17 : tensor<4x128x8xf32>
	}
}
