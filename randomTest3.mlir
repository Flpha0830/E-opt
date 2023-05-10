module {
	func.func @test_random_3() ->tensor<1x8x256xf32> {
		%input8 =  "tosa.const"() {value = dense<9.0>: tensor<1x8x256xf32>} : () -> tensor<1x8x256xf32>
		%input2 =  "tosa.const"() {value = dense<4.0>: tensor<1x8x256xf32>} : () -> tensor<1x8x256xf32>
		%Add0 =  "tosa.add"(%input8, %input2) : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Mul2 =  "tosa.mul"(%input8, %Add0) {shift = 0 : i32} : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Add1 =  "tosa.add"(%input8, %input8) : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Mul6 =  "tosa.mul"(%Add1, %input8) {shift = 0 : i32} : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Add10 =  "tosa.add"(%Mul6, %Add1) : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Mul18 =  "tosa.mul"(%Mul2, %Add10) {shift = 0 : i32} : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		return %Mul18 : tensor<1x8x256xf32>
	}
}
