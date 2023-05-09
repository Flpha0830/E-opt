module {
	func.func @test_random_7() ->tensor<2x8x16xf32> {
		%input9 =  "tosa.const"() {value = dense<2.0>: tensor<2x8x16xf32>} : () -> tensor<2x8x16xf32>
		%input4 =  "tosa.const"() {value = dense<5.0>: tensor<2x16x8xf32>} : () -> tensor<2x16x8xf32>
		%Mul0 =  "tosa.mul"(%input4, %input4) {shift = 0 : i32} : (tensor<2x16x8xf32>, tensor<2x16x8xf32>) -> tensor<2x16x8xf32>
		%Mul1 =  "tosa.mul"(%input4, %Mul0) {shift = 0 : i32} : (tensor<2x16x8xf32>, tensor<2x16x8xf32>) -> tensor<2x16x8xf32>
		%Add6 =  "tosa.add"(%Mul1, %input4) : (tensor<2x16x8xf32>, tensor<2x16x8xf32>) -> tensor<2x16x8xf32>
		%input0 =  "tosa.const"() {value = dense<2.0>: tensor<2x16x16xf32>} : () -> tensor<2x16x16xf32>
		%MatMul3 =  "tosa.matmul"(%input9, %input0) : (tensor<2x8x16xf32>, tensor<2x16x16xf32>) -> tensor<2x8x16xf32>
		%MatMul12 =  "tosa.matmul"(%Add6, %MatMul3) : (tensor<2x16x8xf32>, tensor<2x8x16xf32>) -> tensor<2x16x16xf32>
		%MatMul15 =  "tosa.matmul"(%input9, %MatMul12) : (tensor<2x8x16xf32>, tensor<2x16x16xf32>) -> tensor<2x8x16xf32>
		return %MatMul15 : tensor<2x8x16xf32>
	}
}
