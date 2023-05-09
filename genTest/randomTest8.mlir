module {
	func.func @test_random_8() ->tensor<1x8x256xf32> {
		%input5 =  "tosa.const"() {value = dense<1.0>: tensor<1x8x64xf32>} : () -> tensor<1x8x64xf32>
		%Add5 =  "tosa.add"(%input5, %input5) : (tensor<1x8x64xf32>, tensor<1x8x64xf32>) -> tensor<1x8x64xf32>
		%input0 =  "tosa.const"() {value = dense<0.0>: tensor<1x64x256xf32>} : () -> tensor<1x64x256xf32>
		%MatMul8 =  "tosa.matmul"(%Add5, %input0) : (tensor<1x8x64xf32>, tensor<1x64x256xf32>) -> tensor<1x8x256xf32>
		%input8 =  "tosa.const"() {value = dense<5.0>: tensor<1x64x256xf32>} : () -> tensor<1x64x256xf32>
		%Mul3 =  "tosa.mul"(%input8, %input8) {shift = 0 : i32} : (tensor<1x64x256xf32>, tensor<1x64x256xf32>) -> tensor<1x64x256xf32>
		%MatMul4 =  "tosa.matmul"(%input5, %Mul3) : (tensor<1x8x64xf32>, tensor<1x64x256xf32>) -> tensor<1x8x256xf32>
		%Mul11 =  "tosa.mul"(%MatMul8, %MatMul4) {shift = 0 : i32} : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Mul16 =  "tosa.mul"(%Mul11, %MatMul4) {shift = 0 : i32} : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		%Add17 =  "tosa.add"(%Mul11, %Mul16) : (tensor<1x8x256xf32>, tensor<1x8x256xf32>) -> tensor<1x8x256xf32>
		return %Add17 : tensor<1x8x256xf32>
	}
}
