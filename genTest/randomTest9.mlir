module {
	func.func @test_random_9() ->tensor<2x128x4xf32> {
		%input6 =  "tosa.const"() {value = dense<4.0>: tensor<2x128x4xf32>} : () -> tensor<2x128x4xf32>
		%input4 =  "tosa.const"() {value = dense<6.0>: tensor<2x128x4xf32>} : () -> tensor<2x128x4xf32>
		%Mul1 =  "tosa.mul"(%input6, %input4) {shift = 0 : i32} : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Add4 =  "tosa.add"(%input4, %Mul1) : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Add5 =  "tosa.add"(%input4, %Add4) : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%input2 =  "tosa.const"() {value = dense<5.0>: tensor<2x4x128xf32>} : () -> tensor<2x4x128xf32>
		%MatMul2 =  "tosa.matmul"(%Mul1, %input2) : (tensor<2x128x4xf32>, tensor<2x4x128xf32>) -> tensor<2x128x128xf32>
		%MatMul3 =  "tosa.matmul"(%MatMul2, %input4) : (tensor<2x128x128xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Mul6 =  "tosa.mul"(%MatMul3, %Mul1) {shift = 0 : i32} : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Add11 =  "tosa.add"(%Add5, %Mul6) : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Add14 =  "tosa.add"(%input6, %Add11) : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Mul12 =  "tosa.mul"(%Add11, %input4) {shift = 0 : i32} : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Add23 =  "tosa.add"(%Mul12, %Add4) : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		%Mul24 =  "tosa.mul"(%Add14, %Add23) {shift = 0 : i32} : (tensor<2x128x4xf32>, tensor<2x128x4xf32>) -> tensor<2x128x4xf32>
		return %Mul24 : tensor<2x128x4xf32>
	}
}
