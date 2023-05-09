module {
	func.func @test_random_1() ->tensor<1x16x4xf32> {
		%input9 =  "tosa.const"() {value = dense<6.0>: tensor<1x16x64xf32>} : () -> tensor<1x16x64xf32>
		%Mul2 =  "tosa.mul"(%input9, %input9) {shift = 0 : i32} : (tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
		%input0 =  "tosa.const"() {value = dense<0.0>: tensor<1x16x64xf32>} : () -> tensor<1x16x64xf32>
		%Add3 =  "tosa.add"(%Mul2, %input0) : (tensor<1x16x64xf32>, tensor<1x16x64xf32>) -> tensor<1x16x64xf32>
		%input8 =  "tosa.const"() {value = dense<3.0>: tensor<1x64x4xf32>} : () -> tensor<1x64x4xf32>
		%MatMul6 =  "tosa.matmul"(%Add3, %input8) : (tensor<1x16x64xf32>, tensor<1x64x4xf32>) -> tensor<1x16x4xf32>
		%input6 =  "tosa.const"() {value = dense<7.0>: tensor<1x16x4xf32>} : () -> tensor<1x16x4xf32>
		%Mul9 =  "tosa.mul"(%MatMul6, %input6) {shift = 0 : i32} : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		%Add11 =  "tosa.add"(%Mul9, %MatMul6) : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		%MatMul1 =  "tosa.matmul"(%input9, %input8) : (tensor<1x16x64xf32>, tensor<1x64x4xf32>) -> tensor<1x16x4xf32>
		%Mul13 =  "tosa.mul"(%Add11, %MatMul1) {shift = 0 : i32} : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		%Mul19 =  "tosa.mul"(%Mul13, %MatMul6) {shift = 0 : i32} : (tensor<1x16x4xf32>, tensor<1x16x4xf32>) -> tensor<1x16x4xf32>
		return %Mul19 : tensor<1x16x4xf32>
	}
}
