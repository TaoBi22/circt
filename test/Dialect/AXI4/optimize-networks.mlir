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

// Logic outside the network is left alone, dead or not
// CHECK-LABEL: hw.module @UnrelatedLogic
hw.module @UnrelatedLogic(in %a : i8, out o : i8) {
  // CHECK-NEXT: comb.xor %a, %a
  %dead = comb.xor %a, %a : i8
  hw.output %a : i8
}
