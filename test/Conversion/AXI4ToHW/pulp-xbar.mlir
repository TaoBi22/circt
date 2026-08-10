// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 2>
!sub_lo = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x0, last = 0x7ff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>
!sub_hi = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 5, user_width = 0, windows = <<base = 0x800, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 8, outstanding_reads = 8>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Low(in %axi : !sub_lo)
hw.module.extern @High(in %axi : !sub_hi)

// The external module is unchanged by the option - the wrapper is verilog
// hanging off it, in its own file
// CHECK:       hw.module.extern @axi_xbar_2u2d_a32_d64_i4_o5(
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out sub1_rready : i1)
// CHECK-SAME:    attributes {source = @axi_xbar_2u2d_a32_d64_i4_o5.sv}
// CHECK:      sv.verbatim.source @axi_xbar_2u2d_a32_d64_i4_o5.sv

// The wrapper declares a port per exploded signal, with the same name and the
// mirrored direction
// CHECK-SAME:   module axi_xbar_2u2d_a32_d64_i4_o5 (\0A
// CHECK-SAME:     input  logic clk_i,\0A
// CHECK-SAME:     input  logic rst_ni,\0A
// CHECK-SAME:     input  axi_xbar_2u2d_a32_d64_i4_o5_mgr_aw_t mgr0_aw,\0A
// CHECK-SAME:     input  logic mgr0_awvalid,\0A
// CHECK-SAME:     output logic mgr0_awready,\0A
// CHECK-SAME:     input  axi_xbar_2u2d_a32_d64_i4_o5_mgr_aw_t mgr1_aw,\0A
// CHECK-SAME:     output axi_xbar_2u2d_a32_d64_i4_o5_sub_aw_t sub0_aw,\0A
// CHECK-SAME:     input  logic sub1_rvalid,\0A
// CHECK-SAME:     output logic sub1_rready\0A);

// Two managers need one tag bit downstream, so the crossbar widens the ID -
// the o5 in the module name above
// CHECK-SAME:   NoSlvPorts:         2,\0A
// CHECK-SAME:   NoMstPorts:         2,\0A

// The outstanding counts come from the ports, per side
// CHECK-SAME:   MaxSlvTrans:        4,\0A
// CHECK-SAME:   MaxMstTrans:        8,\0A
// CHECK-SAME:   NoAddrRules:        2,\0A

// One rule per window of each downstream port, half open, indexed by port
// CHECK-SAME:   AddrMap = '{\0A
// CHECK-SAME:     '{idx: 0, start_addr: 32'h0, end_addr: 32'h800},\0A
// CHECK-SAME:     '{idx: 1, start_addr: 32'h800, end_addr: 32'h1000}\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_xbar_2u2d_a32_d64_i4_o5.sv">
// CHECK-SAME:  verilogName = "axi_xbar_2u2d_a32_d64_i4_o5"

// CHECK-LABEL: hw.module @TwoToTwo(
// CHECK:         hw.instance "xbar0" @axi_xbar_2u2d_a32_d64_i4_o5(
hw.module @TwoToTwo(in %clk : !seq.clock, in %rst_ni : i1) {
  %a = hw.instance "mgr_a" @Manager() -> (axi: !mgr)
  %b = hw.instance "mgr_b" @Manager() -> (axi: !mgr)
  %lo, %hi = axi4.xbar %clk, %rst_ni mgrs %a, %b : (!mgr, !mgr) -> (!sub_lo, !sub_hi)
  hw.instance "lo" @Low(axi: %lo: !sub_lo) -> ()
  hw.instance "hi" @High(axi: %hi: !sub_hi) -> ()
}

// -----

// A port with no user field neither reads nor drives PULP's, which is still a
// bit wide, so the crossbar side is tied off
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Subordinate(in %axi : !sub)

// CHECK:      sv.verbatim.source
// CHECK-SAME:   typedef logic [1-1:0] axi_xbar_1u1d_a32_d64_i4_o4_user_t;\0A
// CHECK-SAME:   assign slv_req[0].aw = '{id: mgr0_aw.id,
// CHECK-SAME:     atop: mgr0_aw.atop, user: '0};\0A
// CHECK-SAME:   assign mgr0_b = '{id: slv_resp[0].b.id, resp: slv_resp[0].b.resp};\0A
hw.module @NoUser(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// With a user field both sides carry it
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 3, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 3, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Subordinate(in %axi : !sub)

// CHECK:      sv.verbatim.source
// CHECK-SAME:   typedef logic [3-1:0] axi_xbar_1u1d_a32_d64_i4_o4_user_t;\0A
// CHECK-SAME:   assign slv_req[0].aw = '{id: mgr0_aw.id,
// CHECK-SAME:     atop: mgr0_aw.atop, user: mgr0_aw.user};\0A
// CHECK-SAME:   assign mgr0_b = '{id: slv_resp[0].b.id, resp: slv_resp[0].b.resp, user: slv_resp[0].b.user};\0A
hw.module @WithUser(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}

// -----

// A window reaching the top of the address space wraps to zero, which is how
// PULP describes address windows that run to the end of the address space
!mgr = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xffffffff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!sub = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xffffffff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !mgr)
hw.module.extern @Subordinate(in %axi : !sub)

// CHECK:      sv.verbatim.source
// CHECK-SAME:   '{idx: 0, start_addr: 32'h0, end_addr: 32'h0}\0A
hw.module @WholeSpace(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !mgr)
  %s = axi4.xbar %clk, %rst_ni mgrs %m : (!mgr) -> (!sub)
  hw.instance "sub" @Subordinate(axi: %s: !sub) -> ()
}
