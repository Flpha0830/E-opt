module {
	func.func @test_random_2() ->tensor<2x64x4xf32> {
		%input3 =  "tosa.const"() {value = dense<9.0>: tensor<2x64x4xf32>} : () -> tensor<2x64x4xf32>
		%input5 =  "tosa.const"() {value = dense<9.0>: tensor<2x64x4xf32>} : () -> tensor<2x64x4xf32>
		%Add1 =  "tosa.add"(%input3, %input5) : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Add2 =  "tosa.add"(%Add1, %input3) : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Add5 =  "tosa.add"(%Add2, %Add1) : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Mul0 =  "tosa.mul"(%input5, %input5) {shift = 0 : i32} : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Mul4 =  "tosa.mul"(%input3, %Mul0) {shift = 0 : i32} : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Mul6 =  "tosa.mul"(%Add5, %Mul4) {shift = 0 : i32} : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		%Mul17 =  "tosa.mul"(%Mul6, %Mul0) {shift = 0 : i32} : (tensor<2x64x4xf32>, tensor<2x64x4xf32>) -> tensor<2x64x4xf32>
		return %Mul17 : tensor<2x64x4xf32>
	}
}
