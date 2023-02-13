# circt-lec builds if a logic backend is found
if(CIRCTSMT_ENABLED)

  list(APPEND CIRCT_MC_LIBS
    CIRCTSMT
  )

  add_llvm_tool(circt-mc circt-mc.cpp)
  target_link_libraries(circt-mc PRIVATE ${CIRCT_MC_LIBS})
  target_include_directories(circt-lec PRIVATE ${CIRCT_Z3_INCLUDES})

  # Correct the runpath when linking shared libraries.
  if(BUILD_SHARED_LIBS)
    set_target_properties(circt-mc PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )
  endif()

  llvm_update_compile_flags(circt-mc)
  mlir_check_all_link_libraries(circt-mc)
endif()
