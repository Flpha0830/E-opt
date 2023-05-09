module {
	func.func @test_random_0() ->tensor<2x16x64xf32> {
		%input9 =  "tosa.const"() {value = dense<2.0>: tensor<2x16x64xf32>} : () -> tensor<2x16x64xf32>
		%input7 =  "tosa.const"() {value = dense<1.0>: tensor<2x16x64xf32>} : () -> tensor<2x16x64xf32>
		%Add0 =  "tosa.add"(%input7, %input9) : (tensor<2x16x64xf32>, tensor<2x16x64xf32>) -> tensor<2x16x64xf32>
		%Mul3 =  "tosa.mul"(%input9, %Add0) {shift = 0 : i32} : (tensor<2x16x64xf32>, tensor<2x16x64xf32>) -> tensor<2x16x64xf32>
		%Add9 =  "tosa.add"(%Mul3, %input7) : (tensor<2x16x64xf32>, tensor<2x16x64xf32>) -> tensor<2x16x64xf32>
		%Mul10 =  "tosa.mul"(%Mul3, %Add9) {shift = 0 : i32} : (tensor<2x16x64xf32>, tensor<2x16x64xf32>) -> tensor<2x16x64xf32>
		return %Mul10 : tensor<2x16x64xf32>
	}
}
