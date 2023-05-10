module {
	func.func @test_random_4() ->tensor<4x8x64xf32> {
		%input4 =  "tosa.const"() {value = dense<8.0>: tensor<4x8x64xf32>} : () -> tensor<4x8x64xf32>
		%Add2 =  "tosa.add"(%input4, %input4) : (tensor<4x8x64xf32>, tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
		%Mul4 =  "tosa.mul"(%Add2, %Add2) {shift = 0 : i32} : (tensor<4x8x64xf32>, tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
		%Add6 =  "tosa.add"(%input4, %Mul4) : (tensor<4x8x64xf32>, tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
		%input0 =  "tosa.const"() {value = dense<3.0>: tensor<4x8x64xf32>} : () -> tensor<4x8x64xf32>
		%Add11 =  "tosa.add"(%Add6, %input0) : (tensor<4x8x64xf32>, tensor<4x8x64xf32>) -> tensor<4x8x64xf32>
		return %Add11 : tensor<4x8x64xf32>
	}
}
