module {
	func.func @test_random_0() ->tensor<1x256x16xf32> {
		%input1 =  "tosa.const"() {value = dense<7.0>: tensor<1x256x16xf32>} : () -> tensor<1x256x16xf32>
		%input9 =  "tosa.const"() {value = dense<1.0>: tensor<1x256x16xf32>} : () -> tensor<1x256x16xf32>
		%Mul0 =  "tosa.mul"(%input1, %input9) {shift = 0 : i32} : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		%Mul1 =  "tosa.mul"(%input1, %Mul0) {shift = 0 : i32} : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		%Add3 =  "tosa.add"(%Mul1, %input9) : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		%Mul14 =  "tosa.mul"(%Add3, %Mul0) {shift = 0 : i32} : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		%Mul17 =  "tosa.mul"(%Mul1, %Mul14) {shift = 0 : i32} : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		%Mul16 =  "tosa.mul"(%Add3, %Mul1) {shift = 0 : i32} : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		%Mul18 =  "tosa.mul"(%Mul17, %Mul16) {shift = 0 : i32} : (tensor<1x256x16xf32>, tensor<1x256x16xf32>) -> tensor<1x256x16xf32>
		return %Mul18 : tensor<1x256x16xf32>
	}
}
