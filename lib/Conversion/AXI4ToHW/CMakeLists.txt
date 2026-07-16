add_circt_conversion_library(CIRCTAXI4ToHW
  AXI4ToHW.cpp
  PULPMapping.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTAXI4
  CIRCTHW
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
