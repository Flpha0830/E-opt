module {
	func.func @test_random_6() ->tensor<2x128x128xf32> {
		%input7 =  "tosa.const"() {value = dense<1.0>: tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
		%input0 =  "tosa.const"() {value = dense<7.0>: tensor<2x128x128xf32>} : () -> tensor<2x128x128xf32>
		%Mul4 =  "tosa.mul"(%input7, %input0) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
		%MatMul7 =  "tosa.matmul"(%Mul4, %input7) : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
		%Mul9 =  "tosa.mul"(%MatMul7, %input0) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
		%Mul11 =  "tosa.mul"(%input0, %Mul9) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
		%Mul19 =  "tosa.mul"(%MatMul7, %Mul11) {shift = 0 : i32} : (tensor<2x128x128xf32>, tensor<2x128x128xf32>) -> tensor<2x128x128xf32>
		return %Mul19 : tensor<2x128x128xf32>
	}
}
