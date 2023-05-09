module {
	func.func @test_random_4() ->tensor<1x256x64xf32> {
		%input1 =  "tosa.const"() {value = dense<5.0>: tensor<1x256x64xf32>} : () -> tensor<1x256x64xf32>
		%input6 =  "tosa.const"() {value = dense<4.0>: tensor<1x256x8xf32>} : () -> tensor<1x256x8xf32>
		%input4 =  "tosa.const"() {value = dense<10.0>: tensor<1x8x256xf32>} : () -> tensor<1x8x256xf32>
		%MatMul1 =  "tosa.matmul"(%input4, %input1) : (tensor<1x8x256xf32>, tensor<1x256x64xf32>) -> tensor<1x8x64xf32>
		%MatMul3 =  "tosa.matmul"(%input6, %MatMul1) : (tensor<1x256x8xf32>, tensor<1x8x64xf32>) -> tensor<1x256x64xf32>
		%Add5 =  "tosa.add"(%input1, %MatMul3) : (tensor<1x256x64xf32>, tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
		%Add9 =  "tosa.add"(%input1, %Add5) : (tensor<1x256x64xf32>, tensor<1x256x64xf32>) -> tensor<1x256x64xf32>
		return %Add9 : tensor<1x256x64xf32>
	}
}
