module {
	func.func @test_random_6() ->tensor<1x8x16xf32> {
		%input2 =  "tosa.const"() {value = dense<7.0>: tensor<1x8x16xf32>} : () -> tensor<1x8x16xf32>
		%input7 =  "tosa.const"() {value = dense<7.0>: tensor<1x8x4xf32>} : () -> tensor<1x8x4xf32>
		%input5 =  "tosa.const"() {value = dense<2.0>: tensor<1x4x8xf32>} : () -> tensor<1x4x8xf32>
		%MatMul1 =  "tosa.matmul"(%input7, %input5) : (tensor<1x8x4xf32>, tensor<1x4x8xf32>) -> tensor<1x8x8xf32>
		%MatMul4 =  "tosa.matmul"(%MatMul1, %input2) : (tensor<1x8x8xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Mul6 =  "tosa.mul"(%input2, %MatMul4) {shift = 0 : i32} : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Mul7 =  "tosa.mul"(%Mul6, %input2) {shift = 0 : i32} : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Add8 =  "tosa.add"(%input2, %Mul6) : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Mul10 =  "tosa.mul"(%MatMul4, %Add8) {shift = 0 : i32} : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Mul11 =  "tosa.mul"(%Add8, %Mul10) {shift = 0 : i32} : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Add13 =  "tosa.add"(%input2, %Mul11) : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Add20 =  "tosa.add"(%Add13, %Add8) : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		%Add21 =  "tosa.add"(%Mul7, %Add20) : (tensor<1x8x16xf32>, tensor<1x8x16xf32>) -> tensor<1x8x16xf32>
		return %Add21 : tensor<1x8x16xf32>
	}
}
