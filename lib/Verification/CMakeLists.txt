if(CIRCT_LEC_ENABLED)
  add_circt_library(CIRCTSMT
    Solver.cpp
    Circuit.cpp
    LogicExporter.cpp

    ADDITIONAL_HEADER_DIRS
    ${CIRCT_Z3_INCLUDE_DIR}

    LINK_COMPONENTS
    Core
    
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
