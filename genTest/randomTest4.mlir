module {
	func.func @test_random_4() ->tensor<4x4x32xf32> {
		%input2 =  "tosa.const"() {value = dense<4.0>: tensor<4x4x256xf32>} : () -> tensor<4x4x256xf32>
		%input4 =  "tosa.const"() {value = dense<7.0>: tensor<4x256x256xf32>} : () -> tensor<4x256x256xf32>
		%input6 =  "tosa.const"() {value = dense<4.0>: tensor<4x256x16xf32>} : () -> tensor<4x256x16xf32>
		%input8 =  "tosa.const"() {value = dense<7.0>: tensor<4x16x128xf32>} : () -> tensor<4x16x128xf32>
		%input7 =  "tosa.const"() {value = dense<8.0>: tensor<4x128x16xf32>} : () -> tensor<4x128x16xf32>
		%input3 =  "tosa.const"() {value = dense<2.0>: tensor<4x16x32xf32>} : () -> tensor<4x16x32xf32>
		%MatMul4 =  "tosa.matmul"(%input7, %input3) : (tensor<4x128x16xf32>, tensor<4x16x32xf32>) -> tensor<4x128x32xf32>
		%MatMul5 =  "tosa.matmul"(%input8, %MatMul4) : (tensor<4x16x128xf32>, tensor<4x128x32xf32>) -> tensor<4x16x32xf32>
		%MatMul8 =  "tosa.matmul"(%input6, %MatMul5) : (tensor<4x256x16xf32>, tensor<4x16x32xf32>) -> tensor<4x256x32xf32>
		%MatMul15 =  "tosa.matmul"(%input4, %MatMul8) : (tensor<4x256x256xf32>, tensor<4x256x32xf32>) -> tensor<4x256x32xf32>
		%MatMul16 =  "tosa.matmul"(%input2, %MatMul15) : (tensor<4x4x256xf32>, tensor<4x256x32xf32>) -> tensor<4x4x32xf32>
		return %MatMul16 : tensor<4x4x32xf32>
	}
}
