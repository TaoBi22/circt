if(CIRCT_LEC_ENABLED)
  add_circt_library(CIRCTSMT
    Solver.cpp
    Circuit.cpp
    LogicExporter.cpp

    LINK_COMPONENTS
    Core

    ADDITIONAL_HEADER_DIRS
    ${CIRCT_MAIN_INCLUDE_DIR}/circt/Verification

    LINK_LIBS
    ${CIRCT_Z3_LIBS}
    
    LINK_LIBS PUBLIC
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

  )
endif()
