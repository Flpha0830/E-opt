module {
	func.func @test_random_1() ->tensor<2x32x64xf32> {
		%input3 =  "tosa.const"() {value = dense<4.0>: tensor<2x32x32xf32>} : () -> tensor<2x32x32xf32>
		%input4 =  "tosa.const"() {value = dense<7.0>: tensor<2x32x64xf32>} : () -> tensor<2x32x64xf32>
		%Mul2 =  "tosa.mul"(%input4, %input4) {shift = 0 : i32} : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%MatMul3 =  "tosa.matmul"(%input3, %input4) : (tensor<2x32x32xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Mul4 =  "tosa.mul"(%Mul2, %MatMul3) {shift = 0 : i32} : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Mul6 =  "tosa.mul"(%input4, %Mul4) {shift = 0 : i32} : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%MatMul8 =  "tosa.matmul"(%input3, %Mul6) : (tensor<2x32x32xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%MatMul5 =  "tosa.matmul"(%input3, %MatMul3) : (tensor<2x32x32xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Add11 =  "tosa.add"(%MatMul8, %MatMul5) : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Mul15 =  "tosa.mul"(%Add11, %Mul4) {shift = 0 : i32} : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%MatMul13 =  "tosa.matmul"(%input3, %Mul6) : (tensor<2x32x32xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Add18 =  "tosa.add"(%Mul15, %MatMul13) : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Add9 =  "tosa.add"(%MatMul3, %Mul4) : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		%Mul19 =  "tosa.mul"(%Add18, %Add9) {shift = 0 : i32} : (tensor<2x32x64xf32>, tensor<2x32x64xf32>) -> tensor<2x32x64xf32>
		return %Mul19 : tensor<2x32x64xf32>
	}
}
