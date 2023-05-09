module {
	func.func @test_random_5() ->tensor<1x4x128xf32> {
		%input7 =  "tosa.const"() {value = dense<1.0>: tensor<1x4x128xf32>} : () -> tensor<1x4x128xf32>
		%input1 =  "tosa.const"() {value = dense<0.0>: tensor<1x128x128xf32>} : () -> tensor<1x128x128xf32>
		%MatMul5 =  "tosa.matmul"(%input7, %input1) : (tensor<1x4x128xf32>, tensor<1x128x128xf32>) -> tensor<1x4x128xf32>
		%Mul1 =  "tosa.mul"(%input7, %input7) {shift = 0 : i32} : (tensor<1x4x128xf32>, tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
		%Mul11 =  "tosa.mul"(%Mul1, %Mul1) {shift = 0 : i32} : (tensor<1x4x128xf32>, tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
		%Mul13 =  "tosa.mul"(%MatMul5, %Mul11) {shift = 0 : i32} : (tensor<1x4x128xf32>, tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
		%Mul14 =  "tosa.mul"(%Mul13, %MatMul5) {shift = 0 : i32} : (tensor<1x4x128xf32>, tensor<1x4x128xf32>) -> tensor<1x4x128xf32>
		return %Mul14 : tensor<1x4x128xf32>
	}
}
