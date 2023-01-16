# circt-lec builds if a logic backend is found
if(CIRCT_LEC_ENABLED)

  list(APPEND CIRCT_LEC_LIBS
    CIRCTSMT
    ${CIRCT_Z3_LIBS}
  )

  list(APPEND CIRCT_LEC_INCLUDES
    ${CIRCT_Z3_INCLUDES}
  )

  add_llvm_tool(circt-lec circt-lec.cpp)
  target_link_libraries(circt-lec PRIVATE ${CIRCT_LEC_LIBS})
  target_include_directories(circt-lec PRIVATE ${CIRCT_LEC_INCLUDES})

  # Correct the runpath when linking shared libraries.
  if(BUILD_SHARED_LIBS)
    set_target_properties(circt-lec PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )
  endif()

  llvm_update_compile_flags(circt-lec)
  mlir_check_all_link_libraries(circt-lec)
endif()
