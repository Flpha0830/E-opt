module {
	func.func @test_random_9() ->tensor<1x64x256xf32> {
		%input0 =  "tosa.const"() {value = dense<1.0>: tensor<1x64x256xf32>} : () -> tensor<1x64x256xf32>
		%Add1 =  "tosa.add"(%input0, %input0) : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Mul2 =  "tosa.mul"(%Add1, %input0) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Mul8 =  "tosa.mul"(%input0, %Add1) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Mul9 =  "tosa.mul"(%Add1, %Mul8) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Add11 =  "tosa.add"(%input0, %Mul9) : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Mul12 =  "tosa.mul"(%Mul2, %Add11) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Mul18 =  "tosa.mul"(%Mul12, %Add1) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%Mul19 =  "tosa.mul"(%Mul18, %Mul12) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		return %Mul19 : tensor<1x64x256xf32>
	}
}
