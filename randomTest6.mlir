module {
	func.func @test_random_6() ->tensor<2x64x32xf32> {
		%input7 =  "tosa.const"() {value = dense<8.0>: tensor<2x64x32xf32>} : () -> tensor<2x64x32xf32>
		%input9 =  "tosa.const"() {value = dense<8.0>: tensor<2x64x8xf32>} : () -> tensor<2x64x8xf32>
		%input8 =  "tosa.const"() {value = dense<3.0>: tensor<2x8x32xf32>} : () -> tensor<2x8x32xf32>
		%MatMul0 =  "tosa.matmul"(%input9, %input8) : (tensor<2x64x8xf32>, tensor<2x8x32xf32>) -> tensor<2x64x32xf32>
		%Mul6 =  "tosa.mul"(%MatMul0, %MatMul0) {shift = 0 : i32} : (tensor<2x64x32xf32>, tensor<2x64x32xf32>) -> tensor<2x64x32xf32>
		%Mul9 =  "tosa.mul"(%input7, %Mul6) {shift = 0 : i32} : (tensor<2x64x32xf32>, tensor<2x64x32xf32>) -> tensor<2x64x32xf32>
		%Mul14 =  "tosa.mul"(%input7, %Mul9) {shift = 0 : i32} : (tensor<2x64x32xf32>, tensor<2x64x32xf32>) -> tensor<2x64x32xf32>
		%Add18 =  "tosa.add"(%Mul14, %Mul9) : (tensor<2x64x32xf32>, tensor<2x64x32xf32>) -> tensor<2x64x32xf32>
		return %Add18 : tensor<2x64x32xf32>
	}
}
