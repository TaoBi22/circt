set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(circt-mc
  circt-mc.cpp
)
llvm_update_compile_flags(circt-mc)
target_link_libraries(circt-mc
  PRIVATE
  CIRCTCalyx
  CIRCTESI
  CIRCTFIRRTL
  CIRCTFSM
  CIRCTHandshake
  CIRCTHWArith
  CIRCTInteropDialect
  CIRCTLLHD
  CIRCTMoore
  CIRCTPipelineOps
  CIRCTSSP
  CIRCTSystemC

  MLIRIR
  MLIRParser
)
