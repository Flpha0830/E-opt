module {
	func.func @test_random_7() ->tensor<2x4x256xf32> {
		%input6 =  "tosa.const"() {value = dense<6.0>: tensor<2x4x256xf32>} : () -> tensor<2x4x256xf32>
		%input5 =  "tosa.const"() {value = dense<9.0>: tensor<2x256x8xf32>} : () -> tensor<2x256x8xf32>
		%input0 =  "tosa.const"() {value = dense<8.0>: tensor<2x8x256xf32>} : () -> tensor<2x8x256xf32>
		%Mul1 =  "tosa.mul"(%input0, %input0) {shift = 0 : i32} : (tensor<2x8x256xf32>, tensor<2x8x256xf32>) -> tensor<2x8x256xf32>
		%MatMul2 =  "tosa.matmul"(%input5, %Mul1) : (tensor<2x256x8xf32>, tensor<2x8x256xf32>) -> tensor<2x256x256xf32>
		%MatMul6 =  "tosa.matmul"(%input6, %MatMul2) : (tensor<2x4x256xf32>, tensor<2x256x256xf32>) -> tensor<2x4x256xf32>
		%input8 =  "tosa.const"() {value = dense<1.0>: tensor<2x4x32xf32>} : () -> tensor<2x4x32xf32>
		%Add0 =  "tosa.add"(%input8, %input8) : (tensor<2x4x32xf32>, tensor<2x4x32xf32>) -> tensor<2x4x32xf32>
		%Add4 =  "tosa.add"(%input8, %Add0) : (tensor<2x4x32xf32>, tensor<2x4x32xf32>) -> tensor<2x4x32xf32>
		%input9 =  "tosa.const"() {value = dense<2.0>: tensor<2x4x32xf32>} : () -> tensor<2x4x32xf32>
		%Mul5 =  "tosa.mul"(%Add0, %input9) {shift = 0 : i32} : (tensor<2x4x32xf32>, tensor<2x4x32xf32>) -> tensor<2x4x32xf32>
		%Mul8 =  "tosa.mul"(%Add4, %Mul5) {shift = 0 : i32} : (tensor<2x4x32xf32>, tensor<2x4x32xf32>) -> tensor<2x4x32xf32>
		%input1 =  "tosa.const"() {value = dense<10.0>: tensor<2x32x256xf32>} : () -> tensor<2x32x256xf32>
		%MatMul15 =  "tosa.matmul"(%Mul8, %input1) : (tensor<2x4x32xf32>, tensor<2x32x256xf32>) -> tensor<2x4x256xf32>
		%Mul16 =  "tosa.mul"(%MatMul6, %MatMul15) {shift = 0 : i32} : (tensor<2x4x256xf32>, tensor<2x4x256xf32>) -> tensor<2x4x256xf32>
		%Mul19 =  "tosa.mul"(%Mul16, %MatMul15) {shift = 0 : i32} : (tensor<2x4x256xf32>, tensor<2x4x256xf32>) -> tensor<2x4x256xf32>
		return %Mul19 : tensor<2x4x256xf32>
	}
}
