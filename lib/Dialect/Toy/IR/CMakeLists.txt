add_mlir_dialect_library(MLIRToy
		ToyTypes.cpp
		ToyDialect.cpp
		ToyOps.cpp

        ADDITIONAL_HEADER_DIRS
		../../../../include/Dialect/Toy/IR

        DEPENDS
        MLIRToyOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
