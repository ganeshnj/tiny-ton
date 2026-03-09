# C++ tests (LLVM lit or GoogleTest — configure as needed)
# For now, this is a placeholder.

# Uncomment for LLVM lit-based testing:
# configure_lit_site_cfg(
#   ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
#   ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
#   MAIN_CONFIG
#   ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
# )
#
# add_lit_testsuite(check-ttn "Running tiny-ton regression tests"
#   ${CMAKE_CURRENT_BINARY_DIR}
#   DEPENDS ttc
# )
