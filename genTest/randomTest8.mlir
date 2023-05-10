module {
	func.func @test_random_8() ->tensor<1x16x128xf32> {
		%input0 =  "tosa.const"() {value = dense<4.0>: tensor<1x16x128xf32>} : () -> tensor<1x16x128xf32>
		%Add0 =  "tosa.add"(%input0, %input0) : (tensor<1x16x128xf32>, tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
		%Mul2 =  "tosa.mul"(%Add0, %Add0) {shift = 0 : i32} : (tensor<1x16x128xf32>, tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
		%Add3 =  "tosa.add"(%Add0, %Mul2) : (tensor<1x16x128xf32>, tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
		%Mul10 =  "tosa.mul"(%input0, %Add3) {shift = 0 : i32} : (tensor<1x16x128xf32>, tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
		%Mul16 =  "tosa.mul"(%Mul10, %Mul10) {shift = 0 : i32} : (tensor<1x16x128xf32>, tensor<1x16x128xf32>) -> tensor<1x16x128xf32>
		return %Mul16 : tensor<1x16x128xf32>
	}
}
