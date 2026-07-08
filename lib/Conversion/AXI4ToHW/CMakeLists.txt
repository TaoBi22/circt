add_circt_conversion_library(CIRCTAXI4ToHW
  AXI4ToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTAXI4
  CIRCTHW
  CIRCTComb
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
