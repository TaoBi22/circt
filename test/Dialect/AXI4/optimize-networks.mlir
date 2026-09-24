// RUN: circt-opt %s --optimize-axi4-networks --verify-diagnostics | FileCheck %s

!mgr_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!mgr_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub_gap = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x4000, last = 0x4fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 0, outstanding_reads = 0>

// An abstract subordinate no manager addresses goes, along with the port
// routing to it
// CHECK-LABEL: hw.module @UnreachableSubordinate
hw.module @UnreachableSubordinate(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %[[SUBS:.+]]:2 = axi4.xbar
  // CHECK-SAME: -> (!axi4.port<{{.*}}base = 0x0{{.*}}>, !axi4.port<{{.*}}base = 0x2000{{.*}}>)
  // CHECK-NOT: 0x4000
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  // expected-remark @below {{removed downstream port #2, which no upstream manager addresses}}
  %lo, %hi, %gap = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi, !sub_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo concurrent_writes 4 concurrent_reads 4 : !sub_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi concurrent_writes 4 concurrent_reads 4 : !sub_hi
  axi4.abstract_subordinate %clk, %rst_ni, %gap concurrent_writes 4 concurrent_reads 4 : !sub_gap
}

// The adaptors downstream of it go too
// CHECK-LABEL: hw.module @UnreachableBehindAdaptors
hw.module @UnreachableBehindAdaptors(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %[[SUBS:.+]]:2 = axi4.xbar
  // CHECK-NOT: axi4.cut
  // CHECK-NOT: axi4.id_width_converter
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  // expected-remark @below {{removed downstream port #2, which no upstream manager addresses}}
  %lo, %hi, %gap = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi, !sub_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo concurrent_writes 4 concurrent_reads 4 : !sub_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi concurrent_writes 4 concurrent_reads 4 : !sub_hi
  %cut = axi4.cut %clk, %rst_ni, %gap : !sub_gap
  %converted = axi4.id_width_converter %clk, %rst_ni, %cut
    : (!sub_gap) -> !sub_gap
  axi4.abstract_subordinate %clk, %rst_ni, %converted concurrent_writes 4 concurrent_reads 4 : !sub_gap
}

// A reachable port is left alone however little it is used
// CHECK-LABEL: hw.module @ReachableSubordinate
hw.module @ReachableSubordinate(in %clk : !seq.clock, in %rst_ni : i1) {
  // CHECK: %[[SUBS:.+]]:2 = axi4.xbar
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  %lo, %hi = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi)
  axi4.abstract_subordinate %clk, %rst_ni, %lo concurrent_writes 4 concurrent_reads 4 : !sub_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi concurrent_writes 4 concurrent_reads 4 : !sub_hi
}

// A bridge driving live HW logic is reported rather than removed
// CHECK-LABEL: hw.module @UnreachableLiveBridge
hw.module @UnreachableLiveBridge(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %aw_ready : i1, in %w_ready : i1,
                                 in %b : !hw.struct<id: i5, resp: i2, user: i0>,
                                 in %b_valid : i1, in %ar_ready : i1,
                                 in %r : !hw.struct<id: i5, data: i64, resp: i2, last: i1, user: i0>,
                                 in %r_valid : i1,
                                 out aw_valid : i1) {
  // CHECK: %[[SUBS:.+]]:3 = axi4.xbar
  // CHECK: axi4.port_to_channel_structs
  %mgr_lo = axi4.abstract_manager %clk, %rst_ni : !mgr_lo
  %mgr_hi = axi4.abstract_manager %clk, %rst_ni : !mgr_hi
  // expected-warning @below {{downstream port #2 is not addressed by any upstream manager}}
  %lo, %hi, %gap = axi4.xbar %clk, %rst_ni mgrs %mgr_lo, %mgr_hi
    : (!mgr_lo, !mgr_hi) -> (!sub_lo, !sub_hi, !sub_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo concurrent_writes 4 concurrent_reads 4 : !sub_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi concurrent_writes 4 concurrent_reads 4 : !sub_hi
  // expected-note @+2 {{connected to this operation, which the pass will not remove}}
  %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready =
    axi4.port_to_channel_structs %clk, %rst_ni, %gap
      aw %aw_ready w %w_ready b %b, %b_valid
      ar %ar_ready r %r, %r_valid
 concurrent_writes 4 concurrent_reads 4    : !sub_gap
  hw.output %aw_valid : i1
}

!demuxed = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>, <base = 0x2000, last = 0x2fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!demux_gap = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x4000, last = 0x4fff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 0, outstanding_reads = 0>

// CHECK-LABEL: hw.module @UnreachableDemuxPort
hw.module @UnreachableDemuxPort(in %clk : !seq.clock, in %rst_ni : i1,
                                in %upstream : !demuxed) {
  // CHECK: %[[DOWN:.+]]:2 = axi4.demux
  // CHECK-NOT: 0x4000
  // expected-remark @below {{removed downstream port #2, which no upstream manager addresses}}
  %lo, %hi, %gap = axi4.demux %clk, %rst_ni, %upstream
    : (!demuxed) -> (!mgr_lo, !mgr_hi, !demux_gap)
  axi4.abstract_subordinate %clk, %rst_ni, %lo concurrent_writes 4 concurrent_reads 4 : !mgr_lo
  axi4.abstract_subordinate %clk, %rst_ni, %hi concurrent_writes 4 concurrent_reads 4 : !mgr_hi
  axi4.abstract_subordinate %clk, %rst_ni, %gap concurrent_writes 4 concurrent_reads 4 : !demux_gap
}

//===----------------------------------------------------------------------===//
// Adaptor fusion
//===----------------------------------------------------------------------===//

// Check that adaptors fuse where canonicalization will not: through the cuts
// and crossings in between, and without asking that the conversion invert
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 32>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// Narrowing merges the IDs in flight, so this pair orders transactions the
// fused one leaves free - the same data still arrives
// CHECK-LABEL: hw.module @FuseDippingIdWidths
hw.module @FuseDippingIdWidths(in %clk : !seq.clock, in %rst_ni : i1,
                               in %upstream : !mgr_lo) {
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %upstream
  // CHECK-NOT: axi4.id_width_converter
  %narrow = axi4.id_width_converter %clk, %rst_ni, %upstream
    : (!mgr_lo) -> !narrow_ids
  %wide = axi4.id_width_converter %clk, %rst_ni, %narrow
    : (!narrow_ids) -> !mgr_lo
  axi4.abstract_subordinate %clk, %rst_ni, %wide
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// The cut stays where it was put, but now carries the wider port
// CHECK-LABEL: hw.module @FuseAcrossCut
hw.module @FuseAcrossCut(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[CUT:.+]] = axi4.cut %clk, %rst_ni, %upstream : !axi4.port<{{.*}}data_width = 64,
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %[[CUT]]
  // CHECK-NOT: axi4.data_width_converter
  %narrow = axi4.data_width_converter %clk, %rst_ni, %upstream
    : (!mgr_lo) -> !thin
  %cut = axi4.cut %clk, %rst_ni, %narrow : !thin
  %wide = axi4.data_width_converter %clk, %rst_ni, %cut : (!thin) -> !mgr_lo
  axi4.abstract_subordinate %clk, %rst_ni, %wide
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// A crossing keeps both its clock domains, and widens like a cut
// CHECK-LABEL: hw.module @FuseAcrossCdc
hw.module @FuseAcrossCdc(in %aclk : !seq.clock, in %bclk : !seq.clock,
                         in %rst_ni : i1, in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[CDC:.+]] = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream : !axi4.port<{{.*}}data_width = 64,
  // CHECK-NEXT: axi4.abstract_subordinate %bclk, %rst_ni, %[[CDC]]
  // CHECK-NOT: axi4.data_width_converter
  %narrow = axi4.data_width_converter %aclk, %rst_ni, %upstream
    : (!mgr_lo) -> !thin
  %cdc = axi4.cdc from %aclk to %bclk, %rst_ni, %narrow : !thin
  %wide = axi4.data_width_converter %bclk, %rst_ni, %cdc : (!thin) -> !mgr_lo
  axi4.abstract_subordinate %bclk, %rst_ni, %wide
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// A chain of crossings crosses once, into the domain it ends up in
// CHECK-LABEL: hw.module @FuseCrossingChain
hw.module @FuseCrossingChain(in %aclk : !seq.clock, in %bclk : !seq.clock,
                             in %cclk : !seq.clock, in %rst_ni : i1,
                             in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[CDC:.+]] = axi4.cdc from %aclk to %cclk, %rst_ni, %upstream
  // CHECK-NEXT: axi4.abstract_subordinate %cclk, %rst_ni, %[[CDC]]
  %ab = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream : !mgr_lo
  %bc = axi4.cdc from %bclk to %cclk, %rst_ni, %ab : !mgr_lo
  axi4.abstract_subordinate %cclk, %rst_ni, %bc
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// A chain ending where it started crosses nothing
// CHECK-LABEL: hw.module @FuseCrossingRoundTrip
hw.module @FuseCrossingRoundTrip(in %aclk : !seq.clock, in %bclk : !seq.clock,
                                 in %rst_ni : i1, in %upstream : !mgr_lo) {
  // CHECK-NEXT: axi4.abstract_subordinate %aclk, %rst_ni, %upstream
  // CHECK-NOT: axi4.cdc
  %ab = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream : !mgr_lo
  %ba = axi4.cdc from %bclk to %aclk, %rst_ni, %ab : !mgr_lo
  axi4.abstract_subordinate %aclk, %rst_ni, %ba
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// Fusing across the cut would carry it into the upstream domain
// CHECK-LABEL: hw.module @CrossingsAcrossCut
hw.module @CrossingsAcrossCut(in %aclk : !seq.clock, in %bclk : !seq.clock,
                              in %cclk : !seq.clock, in %rst_ni : i1,
                              in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[AB:.+]] = axi4.cdc from %aclk to %bclk
  // CHECK-NEXT: %[[CUT:.+]] = axi4.cut %bclk, %rst_ni, %[[AB]]
  // CHECK-NEXT: %[[BC:.+]] = axi4.cdc from %bclk to %cclk, %rst_ni, %[[CUT]]
  %ab = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream : !mgr_lo
  %cut = axi4.cut %bclk, %rst_ni, %ab : !mgr_lo
  %bc = axi4.cdc from %bclk to %cclk, %rst_ni, %cut : !mgr_lo
  axi4.abstract_subordinate %cclk, %rst_ni, %bc
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

!mid_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 3, read_id_width = 3, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A fused adaptor takes the PULP config of both adaptors
// CHECK-LABEL: hw.module @FusePulpConfig
hw.module @FusePulpConfig(in %clk : !seq.clock, in %rst_ni : i1,
                          in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[IW:.+]] = axi4.id_width_converter %clk, %rst_ni, %upstream {PULP_CONFIG_AxiMstPortMaxUniqIds = 2 : i32, PULP_CONFIG_AxiSlvPortMaxTxns = 8 : i32}
  // CHECK-NEXT: axi4.abstract_subordinate %clk, %rst_ni, %[[IW]]
  %narrow = axi4.id_width_converter %clk, %rst_ni, %upstream
    {PULP_CONFIG_AxiSlvPortMaxTxns = 8 : i32} : (!mgr_lo) -> !narrow_ids
  %mid = axi4.id_width_converter %clk, %rst_ni, %narrow
    {PULP_CONFIG_AxiMstPortMaxUniqIds = 2 : i32} : (!narrow_ids) -> !mid_ids
  axi4.abstract_subordinate %clk, %rst_ni, %mid
    concurrent_writes 4 concurrent_reads 4 : !mid_ids
}

// A fused crossing takes the PULP config of both crossings
// CHECK-LABEL: hw.module @FuseCrossingPulpConfig
hw.module @FuseCrossingPulpConfig(in %aclk : !seq.clock, in %bclk : !seq.clock,
                                  in %cclk : !seq.clock, in %rst_ni : i1,
                                  in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[CDC:.+]] = axi4.cdc from %aclk to %cclk, %rst_ni, %upstream {PULP_CONFIG_LogDepth = 2 : i32, PULP_CONFIG_SyncStages = 3 : i32}
  // CHECK-NEXT: axi4.abstract_subordinate %cclk, %rst_ni, %[[CDC]]
  %ab = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream
    {PULP_CONFIG_LogDepth = 2 : i32} : !mgr_lo
  %bc = axi4.cdc from %bclk to %cclk, %rst_ni, %ab
    {PULP_CONFIG_SyncStages = 3 : i32} : !mgr_lo
  axi4.abstract_subordinate %cclk, %rst_ni, %bc
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// Crossings setting a PULP parameter to different values are not fused, so
// neither value is lost
// CHECK-LABEL: hw.module @PulpConfigConflict
hw.module @PulpConfigConflict(in %aclk : !seq.clock, in %bclk : !seq.clock,
                              in %cclk : !seq.clock, in %rst_ni : i1,
                              in %upstream : !mgr_lo) {
  // CHECK-NEXT: %[[AB:.+]] = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream {PULP_CONFIG_LogDepth = 2 : i32}
  // CHECK-NEXT: %[[BC:.+]] = axi4.cdc from %bclk to %cclk, %rst_ni, %[[AB]] {PULP_CONFIG_LogDepth = 3 : i32}
  // CHECK-NEXT: axi4.abstract_subordinate %cclk, %rst_ni, %[[BC]]
  %ab = axi4.cdc from %aclk to %bclk, %rst_ni, %upstream
    {PULP_CONFIG_LogDepth = 2 : i32} : !mgr_lo
  %bc = axi4.cdc from %bclk to %cclk, %rst_ni, %ab
    {PULP_CONFIG_LogDepth = 3 : i32} : !mgr_lo
  axi4.abstract_subordinate %cclk, %rst_ni, %bc
    concurrent_writes 4 concurrent_reads 4 : !mgr_lo
}

// Logic outside the network is left alone, dead or not
// CHECK-LABEL: hw.module @UnrelatedLogic
hw.module @UnrelatedLogic(in %a : i8, out o : i8) {
  // CHECK-NEXT: comb.xor %a, %a
  %dead = comb.xor %a, %a : i8
  hw.output %a : i8
}
