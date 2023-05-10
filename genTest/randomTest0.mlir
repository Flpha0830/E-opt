module {
	func.func @test_random_0() ->tensor<4x256x64xf32> {
		%input3 =  "tosa.const"() {value = dense<4.0>: tensor<4x256x64xf32>} : () -> tensor<4x256x64xf32>
		%input5 =  "tosa.const"() {value = dense<8.0>: tensor<4x256x64xf32>} : () -> tensor<4x256x64xf32>
		%Add2 =  "tosa.add"(%input3, %input5) : (tensor<4x256x64xf32>, tensor<4x256x64xf32>) -> tensor<4x256x64xf32>
		%Mul3 =  "tosa.mul"(%input3, %Add2) {shift = 0 : i32} : (tensor<4x256x64xf32>, tensor<4x256x64xf32>) -> tensor<4x256x64xf32>
		%Mul7 =  "tosa.mul"(%Mul3, %input3) {shift = 0 : i32} : (tensor<4x256x64xf32>, tensor<4x256x64xf32>) -> tensor<4x256x64xf32>
		%Add10 =  "tosa.add"(%Mul3, %Mul7) : (tensor<4x256x64xf32>, tensor<4x256x64xf32>) -> tensor<4x256x64xf32>
		%Add12 =  "tosa.add"(%Add10, %input3) : (tensor<4x256x64xf32>, tensor<4x256x64xf32>) -> tensor<4x256x64xf32>
		%Add13 =  "tosa.add"(%Add12, %Mul7) : (tensor<4x256x64xf32>, tensor<4x256x64xf32>) -> tensor<4x256x64xf32>
		return %Add13 : tensor<4x256x64xf32>
	}
}
