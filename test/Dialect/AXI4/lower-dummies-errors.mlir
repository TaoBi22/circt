// RUN: circt-opt %s --lower-axi4-dummies-to-axi --verify-diagnostics

// The pass is not implemented yet, so it rejects any input
// expected-error @below {{lowering the dummies subdialect is not yet implemented}}
module {
  hw.module @NoDummies(in %clk : !seq.clock, in %rst_ni : i1, in %a : i8, out b : i8) {
    hw.output %a : i8
  }
}
