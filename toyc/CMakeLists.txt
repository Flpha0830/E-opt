# This chapter depends on JIT support enabled.
#if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
#    return()
#endif()

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

add_llvm_executable(toyc
  toyc.cpp
  ../lib/Frontend/AST.cpp
  ../lib/CodeGen/MLIRGen.cpp
  ../lib/Dialect/Toy/IR/ToyDialect.cpp
  ../lib/Conversion/LowerToAffineLoops.cpp
  ../lib/Conversion/LowerToLLVM.cpp
  ../lib/Dialect/Toy/Transforms/ShapeInferencePass.cpp
  ../lib/Dialect/Toy/Transforms/ToyCombine.cpp
  ../lib/EGraph/EGraph.cpp
  ../lib/EGraph/EGraphCanonicalizer.cpp
  ../lib/EGraph/EGraphPatternApplicator.cpp

  DEPENDS
  MLIRShapeInferenceInterfaceIncGen
  MLIRToyOpsIncGen
  MLIRToyCombineIncGen
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(toyc
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    )

mlir_check_link_libraries(toyc)
