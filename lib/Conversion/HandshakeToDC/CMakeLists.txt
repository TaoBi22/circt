add_circt_conversion_library(CIRCTHandshakeToDC
    HandshakeToDC.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToDC

    DEPENDS
    CIRCTConversionPassIncGen
    MLIRArithDialect

    LINK_LIBS PUBLIC
    CIRCTHW
    CIRCTHandshake
    CIRCTDC
    MLIRIR
    MLIRPass
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSupport
    MLIRTransforms
)
