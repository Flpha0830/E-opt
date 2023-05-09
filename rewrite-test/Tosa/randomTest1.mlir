module {
	func.func @test_random_1() ->tensor<4x8x256xf32> {
		%input7 =  "tosa.const"() {value = dense<7.0>: tensor<4x8x128xf32>} : () -> tensor<4x8x128xf32>
		%input6 =  "tosa.const"() {value = dense<0.0>: tensor<4x128x256xf32>} : () -> tensor<4x128x256xf32>
		%MatMul17 =  "tosa.matmul"(%input7, %input6) : (tensor<4x8x128xf32>, tensor<4x128x256xf32>) -> tensor<4x8x256xf32>
		%input0 =  "tosa.const"() {value = dense<4.0>: tensor<4x8x128xf32>} : () -> tensor<4x8x128xf32>
		%Mul0 =  "tosa.mul"(%input7, %input0) {shift = 0 : i32} : (tensor<4x8x128xf32>, tensor<4x8x128xf32>) -> tensor<4x8x128xf32>
		%Add1 =  "tosa.add"(%input0, %Mul0) : (tensor<4x8x128xf32>, tensor<4x8x128xf32>) -> tensor<4x8x128xf32>
		%MatMul3 =  "tosa.matmul"(%Add1, %input6) : (tensor<4x8x128xf32>, tensor<4x128x256xf32>) -> tensor<4x8x256xf32>
		%Mul19 =  "tosa.mul"(%MatMul17, %MatMul3) {shift = 0 : i32} : (tensor<4x8x256xf32>, tensor<4x8x256xf32>) -> tensor<4x8x256xf32>
		return %Mul19 : tensor<4x8x256xf32>
	}
}
