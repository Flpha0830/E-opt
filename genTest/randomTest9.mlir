module {
	func.func @test_random_9() ->tensor<4x16x16xf32> {
		%input0 =  "tosa.const"() {value = dense<6.0>: tensor<4x16x16xf32>} : () -> tensor<4x16x16xf32>
		%input8 =  "tosa.const"() {value = dense<6.0>: tensor<4x16x16xf32>} : () -> tensor<4x16x16xf32>
		%Add0 =  "tosa.add"(%input8, %input0) : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%MatMul1 =  "tosa.matmul"(%input0, %Add0) : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%Mul2 =  "tosa.mul"(%MatMul1, %input8) {shift = 0 : i32} : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%MatMul4 =  "tosa.matmul"(%Mul2, %Mul2) : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%MatMul5 =  "tosa.matmul"(%Mul2, %MatMul4) : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%Mul7 =  "tosa.mul"(%MatMul4, %MatMul5) {shift = 0 : i32} : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%Mul9 =  "tosa.mul"(%Mul7, %input0) {shift = 0 : i32} : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%Add6 =  "tosa.add"(%input0, %Mul2) : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		%Add15 =  "tosa.add"(%Mul9, %Add6) : (tensor<4x16x16xf32>, tensor<4x16x16xf32>) -> tensor<4x16x16xf32>
		return %Add15 : tensor<4x16x16xf32>
	}
}
