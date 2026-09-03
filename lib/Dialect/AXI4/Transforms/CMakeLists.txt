add_circt_dialect_library(CIRCTAXI4Transforms
  LowerAXI4DummiesToAXI.cpp
  PruneAXI4Networks.cpp
  VerifyAXI4Networks.cpp

  DEPENDS
  CIRCTAXI4TransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTAXI4
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
)
