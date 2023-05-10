module {
	func.func @test_random_8() ->tensor<1x64x64xf32> {
		%input4 =  "tosa.const"() {value = dense<0.0>: tensor<1x64x128xf32>} : () -> tensor<1x64x128xf32>
		%input6 =  "tosa.const"() {value = dense<8.0>: tensor<1x128x64xf32>} : () -> tensor<1x128x64xf32>
		%MatMul0 =  "tosa.matmul"(%input6, %input4) : (tensor<1x128x64xf32>, tensor<1x64x128xf32>) -> tensor<1x128x128xf32>
		%input0 =  "tosa.const"() {value = dense<4.0>: tensor<1x128x64xf32>} : () -> tensor<1x128x64xf32>
		%Add1 =  "tosa.add"(%input0, %input6) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%MatMul3 =  "tosa.matmul"(%MatMul0, %Add1) : (tensor<1x128x128xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Add2 =  "tosa.add"(%Add1, %input6) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Mul7 =  "tosa.mul"(%input0, %Add2) {shift = 0 : i32} : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Add8 =  "tosa.add"(%MatMul3, %Mul7) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Add12 =  "tosa.add"(%MatMul3, %Add8) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Mul17 =  "tosa.mul"(%Add12, %Add1) {shift = 0 : i32} : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Mul4 =  "tosa.mul"(%input0, %Add1) {shift = 0 : i32} : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Mul9 =  "tosa.mul"(%Add2, %Mul4) {shift = 0 : i32} : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Add10 =  "tosa.add"(%Mul9, %Add8) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Mul14 =  "tosa.mul"(%MatMul3, %Add10) {shift = 0 : i32} : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%Add19 =  "tosa.add"(%Mul17, %Mul14) : (tensor<1x128x64xf32>, tensor<1x128x64xf32>) -> tensor<1x128x64xf32>
		%MatMul22 =  "tosa.matmul"(%input4, %Add19) : (tensor<1x64x128xf32>, tensor<1x128x64xf32>) -> tensor<1x64x64xf32>
		return %MatMul22 : tensor<1x64x64xf32>
	}
}
