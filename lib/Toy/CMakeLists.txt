add_mlir_dialect_library(MLIRToy
		ToyTypes.cpp
        ToyDialect.cpp
		ToyOps.cpp

        ADDITIONAL_HEADER_DIRS
		${PROJECT_SOURCE_DIR}/include/Toy

        DEPENDS
        MLIRToyOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
