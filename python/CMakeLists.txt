include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_toy`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_toy.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ToyPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT ToyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_toy"
  TD_FILE dialects/ToyOps.td
  SOURCES
    dialects/toy.py
  DIALECT_NAME toy)

declare_mlir_python_extension(ToyPythonSources.Extension
  MODULE_NAME _toyDialects
  ADD_TO_PARENT ToyPythonSources
  SOURCES
        ToyExtension.cpp
  EMBED_CAPI_LINK_LIBS
    ToyCAPI
)

################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(ToyPythonCAPI
  INSTALL_COMPONENT ToyPythonModules
  INSTALL_DESTINATION python_packages/toy/mlir_toy/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/toy/mlir_toy/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    ToyPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ToyPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/toy/mlir_toy"
  INSTALL_PREFIX "python_packages/toy/mlir_toy"
  DECLARED_SOURCES
    ToyPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    ToyPythonCAPI
  )
