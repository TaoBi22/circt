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

//===----------------------------------------------------------------------===//
// Chained adaptors
//===----------------------------------------------------------------------===//

// Check that chained adaptors fuse into one converting to the final type, and
// that a fused pair restoring the original type then folds away entirely
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 32>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thinner = !axi4.port<addr_width = 32, data_width = 16, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 64>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @ChainedDataWidthConverters
hw.module @ChainedDataWidthConverters(in %clk : !seq.clock, in %rst_ni : i1,
                                      in %upstream : !port) {
  // CHECK-NEXT: %[[FUSED:.+]] = axi4.data_width_converter %clk, %rst_ni, %upstream : (!axi4.port<addr_width = 32, data_width = 64, {{.*}}) -> !axi4.port<addr_width = 32, data_width = 16,
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %[[FUSED]]
  %narrow = axi4.data_width_converter %clk, %rst_ni, %upstream : (!port) -> !thin
  %narrower = axi4.data_width_converter %clk, %rst_ni, %narrow : (!thin) -> !thinner
  axi4.abstract_subordinate %clk, %rst_ni, %narrower : !thinner
}

// CHECK-LABEL: hw.module @NarrowThenWidenData
hw.module @NarrowThenWidenData(in %clk : !seq.clock, in %rst_ni : i1,
                               in %upstream : !port) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.data_width_converter
  %narrow = axi4.data_width_converter %clk, %rst_ni, %upstream : (!port) -> !thin
  %wide = axi4.data_width_converter %clk, %rst_ni, %narrow : (!thin) -> !port
  axi4.abstract_subordinate %clk, %rst_ni, %wide : !port
}

// CHECK-LABEL: hw.module @WidenThenNarrowData
hw.module @WidenThenNarrowData(in %clk : !seq.clock, in %rst_ni : i1,
                               in %upstream : !thin) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.data_width_converter
  %wide = axi4.data_width_converter %clk, %rst_ni, %upstream : (!thin) -> !port
  %narrow = axi4.data_width_converter %clk, %rst_ni, %wide : (!port) -> !thin
  axi4.abstract_subordinate %clk, %rst_ni, %narrow : !thin
}

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 6, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @WidenThenNarrowIds
hw.module @WidenThenNarrowIds(in %clk : !seq.clock, in %rst_ni : i1,
                              in %upstream : !port) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.id_width_converter
  %wide = axi4.id_width_converter %clk, %rst_ni, %upstream : (!port) -> !wide_ids
  %narrow = axi4.id_width_converter %clk, %rst_ni, %wide : (!wide_ids) -> !port
  axi4.abstract_subordinate %clk, %rst_ni, %narrow : !port
}

!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @ChainedIdWidthConverters
hw.module @ChainedIdWidthConverters(in %clk : !seq.clock, in %rst_ni : i1,
                                    in %upstream : !wide_ids) {
  // CHECK-NEXT: %[[FUSED:.+]] = axi4.id_width_converter %clk, %rst_ni, %upstream : (!axi4.port<{{.*}}write_id_width = 6, {{.*}}) -> !axi4.port<{{.*}}write_id_width = 2,
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %[[FUSED]]
  %narrow = axi4.id_width_converter %clk, %rst_ni, %upstream : (!wide_ids) -> !port
  %narrower = axi4.id_width_converter %clk, %rst_ni, %narrow : (!port) -> !narrow_ids
  axi4.abstract_subordinate %clk, %rst_ni, %narrower : !narrow_ids
}

// Make sure a pair of ID width converters whose narrowest width is in the
// middle isn't merged (as this changes behaviour)
// CHECK-LABEL: hw.module @IdWidthConvertersDippingBelow
hw.module @IdWidthConvertersDippingBelow(in %clk : !seq.clock, in %rst_ni : i1,
                                         in %upstream : !port) {
  // CHECK-NEXT: %[[NARROW:.+]] = axi4.id_width_converter
  // CHECK-NEXT: %[[WIDE:.+]] = axi4.id_width_converter %clk, %rst_ni, %[[NARROW]]
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %[[WIDE]]
  %narrow = axi4.id_width_converter %clk, %rst_ni, %upstream : (!port) -> !narrow_ids
  %wide = axi4.id_width_converter %clk, %rst_ni, %narrow : (!narrow_ids) -> !wide_ids
  axi4.abstract_subordinate %clk, %rst_ni, %wide : !wide_ids
}

// Inherit outstanding request count from final adaptor in pair
!more_reads = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 8>

// CHECK-LABEL: hw.module @ConverterPairChangingOutstanding
hw.module @ConverterPairChangingOutstanding(in %clk : !seq.clock, in %rst_ni : i1,
                                            in %upstream : !port) {
  // CHECK-NEXT: %[[FUSED:.+]] = axi4.data_width_converter %clk, %rst_ni, %upstream : (!axi4.port<{{.*}}outstanding_reads = 4>) -> !axi4.port<{{.*}}outstanding_reads = 8>
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %[[FUSED]]
  %narrow = axi4.data_width_converter %clk, %rst_ni, %upstream : (!port) -> !thin
  %wide = axi4.data_width_converter %clk, %rst_ni, %narrow : (!thin) -> !more_reads
  axi4.abstract_subordinate %clk, %rst_ni, %wide : !more_reads
}

//===----------------------------------------------------------------------===//
// Dead routing ports
//===----------------------------------------------------------------------===//

// Check that downstream ports nothing consumes and no manager can address are
// dropped from the ops routing to them
!mgr_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!mgr_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_gap = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x4000, last = 0x4fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @UnreachableDeadXbarPort
hw.module @UnreachableDeadXbarPort(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %[[SUBS:.+]]:2 = axi4.xbar
  // CHECK-SAME: -> (!axi4.port<{{.*}}base = 0x0{{.*}}>, !axi4.port<{{.*}}base = 0x2000{{.*}}>)
  // CHECK-NOT: 0x4000
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  %lo, %hi, %gap = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi, !sub_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo : !sub_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi : !sub_hi
}

// We don't want to drop ports necessary for full coverage of upstream ports
// CHECK-LABEL: hw.module @ReachableDeadXbarPort
hw.module @ReachableDeadXbarPort(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %[[SUBS:.+]]:2 = axi4.xbar
  // CHECK-SAME: -> (!axi4.port<{{.*}}base = 0x0{{.*}}>, !axi4.port<{{.*}}base = 0x2000{{.*}}>)
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  %lo, %hi = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi)
  axi4.abstract_subordinate %clk, %rst_ni, %lo : !sub_lo
}

// CHECK-LABEL: hw.module @UnreachableLiveXbarPort
hw.module @UnreachableLiveXbarPort(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %[[SUBS:.+]]:3 = axi4.xbar
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  %lo, %hi, %gap = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi, !sub_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo : !sub_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi : !sub_hi
  axi4.abstract_subordinate %clk, %rst_ni, %gap : !sub_gap
}

!demuxed = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!demux_gap = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x4000, last = 0x4fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// CHECK-LABEL: hw.module @UnreachableDeadDemuxPort
hw.module @UnreachableDeadDemuxPort(in %clk : !seq.clock, in %rst_ni : i1,
                                    in %upstream : !demuxed) {
  // CHECK-NEXT: %[[DOWN:.+]]:2 = axi4.demux
  // CHECK-NOT: 0x4000
  %lo, %hi, %gap = axi4.demux %clk, %rst_ni, %upstream
    : (!demuxed) -> (!mgr_lo, !mgr_hi, !demux_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo : !mgr_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi : !mgr_hi
}
