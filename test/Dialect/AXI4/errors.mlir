// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{expected ','}}
"test.attrs"() {a = #axi4.burst_spec<fixed>} : () -> ()

// -----

// expected-error @below {{'fixed' burst 'len' must be between 1 and 16, got 0}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 0>} : () -> ()

// -----

// expected-error @below {{'fixed' burst 'len' must be between 1 and 16, got 17}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 17>} : () -> ()

// -----

// expected-error @below {{'incr' burst 'len' must be between 1 and 256, got 0}}
"test.attrs"() {a = #axi4.burst_spec<incr, len = 0>} : () -> ()

// -----

// expected-error @below {{'incr' burst 'len' must be between 1 and 256, got 257}}
"test.attrs"() {a = #axi4.burst_spec<incr, len = 257>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 1}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 1>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 7}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 7>} : () -> ()

// -----

// expected-error @below {{'wrap' burst 'len' must be 2, 4, 8, or 16, got 32}}
"test.attrs"() {a = #axi4.burst_spec<wrap, len = 32>} : () -> ()

// -----

// expected-error @below {{'burst_set' must be non-empty}}
"test.attrs"() {a = #axi4.burst_set<>} : () -> ()

// -----

// expected-error @below {{window 'last' address 0x3fff must not be less than 'base' address 0x4000}}
"test.attrs"() {a = #axi4.window<base = 0x4000, last = 0x3fff, burst_specs = <<fixed, len = 4>>>} : () -> ()

// -----

// expected-error @below {{'window_set' must be non-empty}}
"test.attrs"() {a = #axi4.window_set<>} : () -> ()

// -----

// expected-error @below {{port 'addr_width' must be at most 64, got 65}}
"test.port"() : () -> !axi4.port<addr_width = 65, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'data_width' must be a power of two between 8 and 1024, got 24}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 24, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'data_width' must be a power of two between 8 and 1024, got 4}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 4, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'data_width' must be a power of two between 8 and 1024, got 2048}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 2048, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'write_id_width' must be at most 32, got 33}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 33, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'read_id_width' must be at most 32, got 33}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 33, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// -----

// expected-error @below {{port 'outstanding_writes' must be at most 4 for a 'write_id_width' of 2, got 5}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 5, outstanding_reads = 4>

// -----

// expected-error @below {{port 'outstanding_reads' must be at most 4 for a 'read_id_width' of 2, got 5}}
"test.port"() : () -> !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 5>

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @Fanout(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.abstract_manager' op port result must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
  %mgr = axi4.abstract_manager %clk, %rst_ni : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr : !port
  axi4.abstract_subordinate %clk, %rst_ni, %mgr : !port
}

// -----

hw.module @NotAPort(in %clk : !seq.clock, in %rst_ni : i1,
                    in %s : !hw.struct<a: i4>, in %v : i1) {
  // expected-error @below {{'port' must be an AXI4 port interface, but got 'i32'}}
  %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid = axi4.channel_structs_to_port %clk, %rst_ni aw %s, %v w %s, %v b %v ar %s, %v r %v : i32
  hw.output
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!bad_aw = !hw.struct<id: i4, addr: i16, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i0>
!w = !hw.struct<data: i64, strb: i8, last: i1, user: i0>
!b = !hw.struct<id: i4, resp: i2, user: i0>
!r = !hw.struct<id: i4, data: i64, resp: i2, last: i1, user: i0>

hw.module @BadPayload(in %clk : !seq.clock, in %rst_ni : i1,
                      in %aw : !bad_aw, in %w : !w, in %v : i1) {
  // expected-error @below {{'axi4.channel_structs_to_port' op failed to verify that AW payload matches the port type}}
  %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid = "axi4.channel_structs_to_port"(%clk, %rst_ni, %aw, %v, %w, %v, %v, %aw, %v, %v) : (!seq.clock, i1, !bad_aw, i1, !w, i1, i1, !bad_aw, i1, i1) -> (!port, i1, i1, !b, i1, i1, !r, i1)
  hw.output
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @NoManagers(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.xbar' op must have at least one upstream port}}
  %sub = axi4.xbar %clk, %rst_ni mgrs : () -> !mgr
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @NoSubordinates(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.xbar' op must have at least one downstream port}}
  axi4.xbar %clk, %rst_ni mgrs %mgr : (!mgr) -> ()
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_mgr = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

hw.module @MismatchedManagers(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = axi4.abstract_manager %clk, %rst_ni : !mgr
  %b = axi4.abstract_manager %clk, %rst_ni : !narrow_mgr
  // expected-error @below {{'axi4.xbar' op upstream port #1's 'data_width' (32) must match upstream port #0's (64)}}
  %sub = axi4.xbar %clk, %rst_ni mgrs %a, %b : (!mgr, !narrow_mgr) -> !sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide_sub = !axi4.port<addr_width = 32, data_width = 128, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ConvertingXbar(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.xbar' op downstream port #0's 'data_width' (128) must match upstream port #0's (64)}}
  %sub = axi4.xbar %clk, %rst_ni mgrs %mgr : (!mgr) -> !wide_sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

hw.module @NarrowIds(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = axi4.abstract_manager %clk, %rst_ni : !mgr
  %b = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.xbar' op downstream port #0's 'write_id_width' must be at least 5 to tag transactions from 2 managers, got 4}}
  %sub = axi4.xbar %clk, %rst_ni mgrs %a, %b : (!mgr, !mgr) -> !sub
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!other_sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x2fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @OverlappingSubordinates(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.xbar' op downstream ports #0 and #1 have overlapping windows}}
  %a, %b = axi4.xbar %clk, %rst_ni mgrs %mgr : (!mgr) -> (!sub, !other_sub)
}

// -----

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnroutedWindow(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.xbar' op address 0x1000, in upstream port #0's windows, is not covered by any downstream port}}
  %sub = axi4.xbar %clk, %rst_ni mgrs %mgr : (!mgr) -> !sub
}

// -----

// Check a window strictly inside a downstream one still has its bursts checked
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>, <incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnsupportedBurst(in %clk : !seq.clock, in %rst_ni : i1) {
  %mgr = axi4.abstract_manager %clk, %rst_ni : !mgr
  // expected-error @below {{'axi4.xbar' op downstream port #0 does not support all the bursts upstream port #0 issues at address 0x0; upstream requires #axi4.burst_set<<fixed, len = 4>, <incr, len = 8>>, downstream supports #axi4.burst_set<<fixed, len = 4>>}}
  %sub = axi4.xbar %clk, %rst_ni mgrs %mgr : (!mgr) -> !sub
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ConvertingCut(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !port) {
  // expected-error @below {{'axi4.cut' op failed to verify that downstream port must match the upstream port}}
  %cut = "axi4.cut"(%clk, %rst_ni, %upstream) : (!seq.clock, i1, !port) -> !narrow
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ConvertingCdc(in %upstream_clk : !seq.clock,
                         in %downstream_clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !port) {
  // expected-error @below {{'axi4.cdc' op failed to verify that downstream port must match the upstream port}}
  %cdc = "axi4.cdc"(%upstream_clk, %downstream_clk, %rst_ni, %upstream) : (!seq.clock, !seq.clock, i1, !port) -> !narrow
}

// -----

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_addr = !axi4.port<addr_width = 16, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ReaddressingConverter(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %upstream : !wide) {
  // expected-error @below {{'axi4.data_width_converter' op downstream port's 'addr_width' (16) must match upstream port's (32)}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!wide) -> !narrow_addr
}

// -----

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!split = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x7ff, burst_specs = <<fixed, len = 8>>>, <base = 0x800, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @SplittingConverter(in %clk : !seq.clock, in %rst_ni : i1,
                              in %upstream : !wide) {
  // expected-error @below {{'axi4.data_width_converter' op upstream and downstream windows must cover the same addresses}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!wide) -> !split
}

// -----

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!moved = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @MovingConverter(in %clk : !seq.clock, in %rst_ni : i1,
                           in %upstream : !wide) {
  // expected-error @below {{'axi4.data_width_converter' op upstream and downstream windows must cover the same addresses}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!wide) -> !moved
}

// -----

// A single 32-bit beat cannot be carried in whole 64-bit beats
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @IndivisibleBurst(in %clk : !seq.clock, in %rst_ni : i1,
                            in %upstream : !thin) {
  // expected-error @below {{'axi4.data_width_converter' op upstream burst #axi4.burst_spec<fixed, len = 1> does not divide into whole 64-bit beats}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!thin) -> !wide
}

// -----

// A 16-beat wrap would need 32 beats at half the width, which AXI4 cannot express
!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 16>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnrepresentableBurst(in %clk : !seq.clock, in %rst_ni : i1,
                                in %upstream : !wide) {
  // expected-error @below {{'axi4.data_width_converter' op upstream burst #axi4.burst_spec<wrap, len = 16> has no 32-bit equivalent: 'wrap' burst 'len' must be 2, 4, 8, or 16, got 32}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!wide) -> !thin
}

// -----

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!unscaled = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnscaledBurst(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !wide) {
  // expected-error @below {{'axi4.data_width_converter' op downstream window must support at least #axi4.burst_set<<incr, len = 8>> (the upstream's bursts in beats of 32 bits), but supports #axi4.burst_set<<incr, len = 4>>}}
  %dwc = axi4.data_width_converter %clk, %rst_ni, %upstream : (!wide) -> !unscaled
}

// -----

// An ID width conversion re-tags and nothing else, so every other width
// carries through
!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thin_data = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @RewidthingIdConverter(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %upstream : !wide_ids) {
  // expected-error @below {{'axi4.id_width_converter' op downstream port's 'data_width' (32) must match upstream port's (64)}}
  %iwc = axi4.id_width_converter %clk, %rst_ni, %upstream : (!wide_ids) -> !thin_data
}

// -----

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!moved_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @MovingIdConverter(in %clk : !seq.clock, in %rst_ni : i1,
                             in %upstream : !wide_ids) {
  // expected-error @below {{'axi4.id_width_converter' op upstream and downstream windows must cover the same addresses}}
  %iwc = axi4.id_width_converter %clk, %rst_ni, %upstream : (!wide_ids) -> !moved_ids
}

// -----

// A splitter changes burst lengths and nothing else, so every width carries
// through
!bursty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @NarrowingSplitter(in %clk : !seq.clock, in %rst_ni : i1,
                             in %upstream : !bursty) {
  // expected-error @below {{'axi4.burst_splitter' op downstream port's 'data_width' (32) must match upstream port's (64)}}
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream : (!bursty) -> !narrow
}

// -----

!bursty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!moved = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 1>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @MovingSplitter(in %clk : !seq.clock, in %rst_ni : i1,
                          in %upstream : !bursty) {
  // expected-error @below {{'axi4.burst_splitter' op upstream and downstream windows must cover the same addresses}}
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream : (!bursty) -> !moved
}

// -----

// Splitting a wrap burst leaves incrementing beats, so a downstream port still
// typed to wrap does not match
!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!still_wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 2>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @SplitStillWrapping(in %clk : !seq.clock, in %rst_ni : i1,
                              in %upstream : !wrapping) {
  // expected-error @below {{'axi4.burst_splitter' op downstream window must support at least #axi4.burst_set<<incr, len = 1>> (the upstream's bursts split into single beats), but supports #axi4.burst_set<<wrap, len = 2>>}}
  %split = axi4.burst_splitter %clk, %rst_ni, %upstream : (!wrapping) -> !still_wrapping
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @EmptyDemux(in %clk : !seq.clock, in %rst_ni : i1,
                      in %upstream : !port) {
  // expected-error @below {{'axi4.demux' op must have at least one downstream port}}
  axi4.demux %clk, %rst_ni, %upstream : (!port) -> ()
}

// -----

!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_data = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @RewidthingUnwrapper(in %clk : !seq.clock, in %rst_ni : i1,
                               in %upstream : !wrapping) {
  // expected-error @below {{'axi4.burst_unwrapper' op downstream port's 'data_width' (32) must match upstream port's (64)}}
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream : (!wrapping) -> !narrow_data
}

// -----

!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!moved = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x1fff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @MovingUnwrapper(in %clk : !seq.clock, in %rst_ni : i1,
                           in %upstream : !wrapping) {
  // expected-error @below {{'axi4.burst_unwrapper' op upstream and downstream windows must cover the same addresses}}
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream : (!wrapping) -> !moved
}

// -----

!wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!still_wrapping = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<wrap, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @StillWrapping(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !wrapping) {
  // expected-error @below {{'axi4.burst_unwrapper' op downstream window must support at least #axi4.burst_set<<incr, len = 4>> (the upstream's wrapping bursts as incrementing ones), but supports #axi4.burst_set<<wrap, len = 4>>}}
  %unwrapped = axi4.burst_unwrapper %clk, %rst_ni, %upstream : (!wrapping) -> !still_wrapping
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!tagged = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @TaggingDemux(in %clk : !seq.clock, in %rst_ni : i1,
                        in %upstream : !port) {
  // expected-error @below {{'axi4.demux' op downstream port #0's 'write_id_width' (5) must match upstream port's (4)}}
  %sub = axi4.demux %clk, %rst_ni, %upstream : (!port) -> !tagged
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x1000, last = 0x2fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @OverlappingDemux(in %clk : !seq.clock, in %rst_ni : i1,
                            in %upstream : !port) {
  // expected-error @below {{'axi4.demux' op downstream ports #0 and #1 have overlapping windows}}
  %a, %b = axi4.demux %clk, %rst_ni, %upstream : (!port) -> (!lo, !hi)
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnroutedDemux(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !port) {
  // expected-error @below {{'axi4.demux' op address 0x1000, in upstream port's windows, is not covered by any downstream port}}
  %sub = axi4.demux %clk, %rst_ni, %upstream : (!port) -> !lo
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>, <incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>
!fixed_only = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnsupportedDemuxBurst(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %upstream : !port) {
  // expected-error @below {{'axi4.demux' op downstream port #0 does not support all the bursts upstream port issues at address 0x0; upstream requires #axi4.burst_set<<fixed, len = 4>, <incr, len = 8>>, downstream supports #axi4.burst_set<<fixed, len = 4>>}}
  %sub = axi4.demux %clk, %rst_ni, %upstream : (!port) -> !fixed_only
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @EmptyMux(in %clk : !seq.clock, in %rst_ni : i1) {
  // expected-error @below {{'axi4.mux' op must have at least one upstream port}}
  %sub = "axi4.mux"(%clk, %rst_ni) : (!seq.clock, i1) -> !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_addr = !axi4.port<addr_width = 16, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @MismatchedMuxManagers(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %a : !port, in %b : !narrow_addr) {
  // expected-error @below {{'axi4.mux' op upstream port #1's 'addr_width' (16) must match upstream port #0's (32)}}
  %sub = axi4.mux %clk, %rst_ni, %a, %b : (!port, !narrow_addr) -> !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide_id = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!tagged = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @MismatchedMuxIds(in %clk : !seq.clock, in %rst_ni : i1,
                            in %a : !port, in %b : !wide_id) {
  // expected-error @below {{'axi4.mux' op upstream port #1's 'read_id_width' (5) must match upstream port #0's (4)}}
  %sub = axi4.mux %clk, %rst_ni, %a, %b : (!port, !wide_id) -> !tagged
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @ConvertingMux(in %clk : !seq.clock, in %rst_ni : i1,
                         in %upstream : !thin) {
  // expected-error @below {{'axi4.mux' op downstream port's 'data_width' (64) must match upstream port #0's (32)}}
  %sub = axi4.mux %clk, %rst_ni, %upstream : (!thin) -> !port
}

// -----

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!with_user = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 4, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @DroppingUserMux(in %clk : !seq.clock, in %rst_ni : i1,
                           in %upstream : !with_user) {
  // expected-error @below {{'axi4.mux' op downstream port's 'user_width' (0) must match upstream port #0's (4)}}
  %sub = axi4.mux %clk, %rst_ni, %upstream : (!with_user) -> !port
}

// -----

// A mux tags each manager's transactions with its index, so the downstream IDs
// must be wide enough to carry the tag
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow_tag = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 6, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @NarrowMuxIds(in %clk : !seq.clock, in %rst_ni : i1,
                        in %a : !port, in %b : !port, in %c : !port) {
  // expected-error @below {{'axi4.mux' op downstream port's 'write_id_width' must be at least 6 to tag transactions from 3 managers, got 5}}
  %sub = axi4.mux %clk, %rst_ni, %a, %b, %c : (!port, !port, !port) -> !narrow_tag
}

// -----

!wide_window = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0x1fff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!tagged_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnroutedMux(in %clk : !seq.clock, in %rst_ni : i1,
                       in %a : !lo, in %b : !wide_window) {
  // expected-error @below {{'axi4.mux' op address 0x1000, in upstream port #1's windows, is not covered by any downstream port}}
  %sub = axi4.mux %clk, %rst_ni, %a, %b : (!lo, !wide_window) -> !tagged_lo
}

// -----

!bursty = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>, <incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>
!fixed_only = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @UnsupportedMuxBurst(in %clk : !seq.clock, in %rst_ni : i1,
                               in %upstream : !bursty) {
  // expected-error @below {{'axi4.mux' op downstream port #0 does not support all the bursts upstream port #0 issues at address 0x0; upstream requires #axi4.burst_set<<fixed, len = 4>, <incr, len = 8>>, downstream supports #axi4.burst_set<<fixed, len = 4>>}}
  %sub = axi4.mux %clk, %rst_ni, %upstream : (!bursty) -> !fixed_only
}

// -----

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @NarrowMemReadData(in %clk : !seq.clock, in %rst_ni : i1,
                             in %port : !mem_port, in %v : i1, in %rdata : i32) {
  // expected-error @below {{'axi4.to_mem' op failed to verify that read data is as wide as the port's data}}
  %valid, %addr, %wdata, %strb, %we = "axi4.to_mem"(%clk, %rst_ni, %port, %v, %rdata) : (!seq.clock, i1, !mem_port, i1, i32) -> (i1, i32, i64, i8, i1)
}

// -----

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @WideMemAddress(in %clk : !seq.clock, in %rst_ni : i1,
                          in %port : !mem_port, in %v : i1, in %rdata : i64) {
  // expected-error @below {{'axi4.to_mem' op failed to verify that address is as wide as the port's addresses}}
  %valid, %addr, %wdata, %strb, %we = "axi4.to_mem"(%clk, %rst_ni, %port, %v, %rdata) : (!seq.clock, i1, !mem_port, i1, i64) -> (i1, i64, i64, i8, i1)
}

// -----

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module @NarrowMemWriteData(in %clk : !seq.clock, in %rst_ni : i1,
                              in %port : !mem_port, in %v : i1, in %rdata : i64) {
  // expected-error @below {{'axi4.to_mem' op failed to verify that write data is as wide as the port's data}}
  %valid, %addr, %wdata, %strb, %we = "axi4.to_mem"(%clk, %rst_ni, %port, %v, %rdata) : (!seq.clock, i1, !mem_port, i1, i64) -> (i1, i32, i32, i8, i1)
}

// -----

!mem_port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

// A strobe carries a bit per byte of write data, so it is as wide as the data
// width in bytes rather than the address width in bytes
hw.module @AddressSizedMemStrobe(in %clk : !seq.clock, in %rst_ni : i1,
                                 in %port : !mem_port, in %v : i1, in %rdata : i64) {
  // expected-error @below {{'axi4.to_mem' op failed to verify that strobe has a bit per byte of the port's data}}
  %valid, %addr, %wdata, %strb, %we = "axi4.to_mem"(%clk, %rst_ni, %port, %v, %rdata) : (!seq.clock, i1, !mem_port, i1, i64) -> (i1, i32, i64, i4, i1)
}
