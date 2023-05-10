module {
	func.func @test_random_7() ->tensor<2x16x4xf32> {
		%input2 =  "tosa.const"() {value = dense<4.0>: tensor<2x16x4xf32>} : () -> tensor<2x16x4xf32>
		%input7 =  "tosa.const"() {value = dense<10.0>: tensor<2x16x4xf32>} : () -> tensor<2x16x4xf32>
		%Mul5 =  "tosa.mul"(%input2, %input7) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Add0 =  "tosa.add"(%input7, %input2) : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Add7 =  "tosa.add"(%Add0, %input2) : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Mul9 =  "tosa.mul"(%Mul5, %Add7) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Mul11 =  "tosa.mul"(%Mul9, %Add7) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Mul13 =  "tosa.mul"(%Mul11, %Mul5) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Add1 =  "tosa.add"(%input7, %Add0) : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Add2 =  "tosa.add"(%Add1, %Add0) : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Mul6 =  "tosa.mul"(%Add2, %input2) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Mul14 =  "tosa.mul"(%Add2, %Mul6) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		%Mul15 =  "tosa.mul"(%Mul13, %Mul14) {shift = 0 : i32} : (tensor<2x16x4xf32>, tensor<2x16x4xf32>) -> tensor<2x16x4xf32>
		return %Mul15 : tensor<2x16x4xf32>
	}
}
