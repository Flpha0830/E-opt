module {
	func.func @test_random_4() ->tensor<1x32x128xf32> {
		%input4 =  "tosa.const"() {value = dense<1.0>: tensor<1x32x128xf32>} : () -> tensor<1x32x128xf32>
		%input1 =  "tosa.const"() {value = dense<5.0>: tensor<1x128x128xf32>} : () -> tensor<1x128x128xf32>
		%input7 =  "tosa.const"() {value = dense<2.0>: tensor<1x128x4xf32>} : () -> tensor<1x128x4xf32>
		%MatMul4 =  "tosa.matmul"(%input1, %input7) : (tensor<1x128x128xf32>, tensor<1x128x4xf32>) -> tensor<1x128x4xf32>
		%input2 =  "tosa.const"() {value = dense<3.0>: tensor<1x4x128xf32>} : () -> tensor<1x4x128xf32>
		%MatMul7 =  "tosa.matmul"(%MatMul4, %input2) : (tensor<1x128x4xf32>, tensor<1x4x128xf32>) -> tensor<1x128x128xf32>
		%Mul10 =  "tosa.mul"(%input1, %MatMul7) {shift = 0 : i32} : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
		%Add11 =  "tosa.add"(%input1, %Mul10) : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
		%Mul15 =  "tosa.mul"(%Add11, %Mul10) {shift = 0 : i32} : (tensor<1x128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
		%MatMul20 =  "tosa.matmul"(%input4, %Mul15) : (tensor<1x32x128xf32>, tensor<1x128x128xf32>) -> tensor<1x32x128xf32>
		return %MatMul20 : tensor<1x32x128xf32>
	}
}
