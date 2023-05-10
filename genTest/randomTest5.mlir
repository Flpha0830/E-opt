module {
	func.func @test_random_5() ->tensor<1x8x32xf32> {
		%input3 =  "tosa.const"() {value = dense<8.0>: tensor<1x8x32xf32>} : () -> tensor<1x8x32xf32>
		%input0 =  "tosa.const"() {value = dense<5.0>: tensor<1x8x32xf32>} : () -> tensor<1x8x32xf32>
		%input5 =  "tosa.const"() {value = dense<9.0>: tensor<1x8x32xf32>} : () -> tensor<1x8x32xf32>
		%Mul0 =  "tosa.mul"(%input0, %input5) {shift = 0 : i32} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Add2 =  "tosa.add"(%input3, %Mul0) : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Mul1 =  "tosa.mul"(%input5, %input0) {shift = 0 : i32} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Mul6 =  "tosa.mul"(%input0, %Mul0) {shift = 0 : i32} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Mul8 =  "tosa.mul"(%Mul1, %Mul6) {shift = 0 : i32} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Mul9 =  "tosa.mul"(%Mul8, %Mul6) {shift = 0 : i32} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Add17 =  "tosa.add"(%input5, %Mul9) : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Mul19 =  "tosa.mul"(%Add2, %Add17) {shift = 0 : i32} : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		%Add20 =  "tosa.add"(%Add2, %Mul19) : (tensor<1x8x32xf32>, tensor<1x8x32xf32>) -> tensor<1x8x32xf32>
		return %Add20 : tensor<1x8x32xf32>
	}
}
