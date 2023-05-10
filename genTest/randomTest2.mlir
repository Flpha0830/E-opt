module {
	func.func @test_random_2() ->tensor<4x32x64xf32> {
		%input6 =  "tosa.const"() {value = dense<8.0>: tensor<4x32x64xf32>} : () -> tensor<4x32x64xf32>
		%input7 =  "tosa.const"() {value = dense<1.0>: tensor<4x64x64xf32>} : () -> tensor<4x64x64xf32>
		%MatMul1 =  "tosa.matmul"(%input6, %input7) : (tensor<4x32x64xf32>, tensor<4x64x64xf32>) -> tensor<4x32x64xf32>
		%Mul10 =  "tosa.mul"(%MatMul1, %input6) {shift = 0 : i32} : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%Add11 =  "tosa.add"(%MatMul1, %Mul10) : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%MatMul0 =  "tosa.matmul"(%input6, %input7) : (tensor<4x32x64xf32>, tensor<4x64x64xf32>) -> tensor<4x32x64xf32>
		%Mul4 =  "tosa.mul"(%MatMul1, %MatMul0) {shift = 0 : i32} : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%MatMul3 =  "tosa.matmul"(%MatMul0, %input7) : (tensor<4x32x64xf32>, tensor<4x64x64xf32>) -> tensor<4x32x64xf32>
		%Mul5 =  "tosa.mul"(%Mul4, %MatMul3) {shift = 0 : i32} : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%Mul9 =  "tosa.mul"(%Mul5, %MatMul0) {shift = 0 : i32} : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%Add12 =  "tosa.add"(%Mul4, %MatMul0) : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%Add14 =  "tosa.add"(%Mul9, %Add12) : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%Mul17 =  "tosa.mul"(%Mul9, %Add14) {shift = 0 : i32} : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		%Add22 =  "tosa.add"(%Add11, %Mul17) : (tensor<4x32x64xf32>, tensor<4x32x64xf32>) -> tensor<4x32x64xf32>
		return %Add22 : tensor<4x32x64xf32>
	}
}
