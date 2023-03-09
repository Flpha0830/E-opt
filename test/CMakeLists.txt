configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TOY_TEST_DEPENDS
        FileCheck count not
        toy-opt
        toy-translate
        )
if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND TOY_TEST_DEPENDS ToyPythonModules)
endif()

add_lit_testsuite(check-toy "Running the toy regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TOY_TEST_DEPENDS}
        )
set_target_properties(check-toy PROPERTIES FOLDER "Tests")

add_lit_testsuites(TOY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TOY_TEST_DEPENDS})
