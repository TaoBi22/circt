// RUN: circt-opt %s --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// Identity adaptors
//===----------------------------------------------------------------------===//

// Check that no-op adaptors are canonicalized away
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @IdentityDataWidthConverter
hw.module @IdentityDataWidthConverter(in %clk : !seq.clock, in %rst_ni : i1,
                                      in %upstream : !port) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.data_width_converter
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!port) -> !port
  axi4.abstract_subordinate %clk, %rst_ni, %dwc : !port
}

// CHECK-LABEL: hw.module @IdentityIdWidthConverter
hw.module @IdentityIdWidthConverter(in %clk : !seq.clock, in %rst_ni : i1,
                                    in %upstream : !port) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.id_width_converter
  %iwc = axi4.id_width_converter %clk, %rst_ni, %upstream : (!port) -> !port
  axi4.abstract_subordinate %clk, %rst_ni, %iwc : !port
}

!beats = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @IdempotentBurstSplitter
hw.module @IdempotentBurstSplitter(in %clk : !seq.clock, in %rst_ni : i1,
                                   in %upstream : !beats) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.burst_splitter
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream : (!beats) -> !beats
  axi4.abstract_subordinate %clk, %rst_ni, %split : !beats
}

// CHECK-LABEL: hw.module @IdempotentBurstUnwrapper
hw.module @IdempotentBurstUnwrapper(in %clk : !seq.clock, in %rst_ni : i1,
                                    in %upstream : !port) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.burst_unwrapper
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream : (!port) -> !port
  axi4.abstract_subordinate %clk, %rst_ni, %unwrapped : !port
}

// CHECK-LABEL: hw.module @SameClockCdc
hw.module @SameClockCdc(in %clk : !seq.clock, in %rst_ni : i1,
                        in %upstream : !port) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.cdc
  %cdc = axi4.cdc from %clk to %clk, %rst_ni, %upstream : !port
  axi4.abstract_subordinate %clk, %rst_ni, %cdc : !port
}
