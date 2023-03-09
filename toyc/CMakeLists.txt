# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
    return()
endif()

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

set(LLVM_TARGET_DEFINITIONS ../lib/mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(MLIRToyCombineIncGen)

add_llvm_executable(toyc
  toyc.cpp
  ../parser/AST.cpp
  ../lib/mlir/MLIRGen.cpp
  ../lib/Toy/ToyDialect.cpp
  ../lib/mlir/LowerToAffineLoops.cpp
  ../lib/mlir/LowerToLLVM.cpp
  ../lib/mlir/ShapeInferencePass.cpp
  ../lib/mlir/ToyCombine.cpp

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
