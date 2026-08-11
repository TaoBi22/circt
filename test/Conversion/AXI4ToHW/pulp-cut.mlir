// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !port)

// The external module is unchanged by the option - the wrapper is verilog
// hanging off it, in its own file
// CHECK:       hw.module.extern @axi_cut_a32_d64_i4(
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i4, addr: i32,
// CHECK-SAME:    out sub0_rready : i1)
// CHECK-SAME:    attributes {source = @axi_cut_a32_d64_i4.sv}
// CHECK:      sv.verbatim.source @axi_cut_a32_d64_i4.sv

// The write path carries the write ID and the read path the read one, so that
// the two widths stay independent - axi_cut never inspects them
// CHECK-SAME:   typedef logic [4-1:0] axi_cut_a32_d64_i4_wid_t;\0A
// CHECK-SAME:   typedef logic [4-1:0] axi_cut_a32_d64_i4_rid_t;\0A
// CHECK-SAME:   `AXI_TYPEDEF_AW_CHAN_T(axi_cut_a32_d64_i4_aw_chan_t, axi_cut_a32_d64_i4_addr_t, axi_cut_a32_d64_i4_wid_t, axi_cut_a32_d64_i4_user_t)\0A
// CHECK-SAME:   `AXI_TYPEDEF_AR_CHAN_T(axi_cut_a32_d64_i4_ar_chan_t, axi_cut_a32_d64_i4_addr_t, axi_cut_a32_d64_i4_rid_t, axi_cut_a32_d64_i4_user_t)\0A

// The wrapper declares a port per exploded signal, with the same name and the
// mirrored direction
// CHECK-SAME:   module axi_cut_a32_d64_i4 (\0A
// CHECK-SAME:     input  logic clk_i,\0A
// CHECK-SAME:     input  logic rst_ni,\0A
// CHECK-SAME:     input  axi_cut_a32_d64_i4_mgr_aw_t mgr0_aw,\0A
// CHECK-SAME:     input  logic mgr0_awvalid,\0A
// CHECK-SAME:     output logic mgr0_awready,\0A
// CHECK-SAME:     output axi_cut_a32_d64_i4_sub_aw_t sub0_aw,\0A
// CHECK-SAME:     input  logic sub0_rvalid,\0A
// CHECK-SAME:     output logic sub0_rready\0A);

// One req/resp pair per side, bridged to the ports of that side
// CHECK-SAME:   axi_cut_a32_d64_i4_req_t  slv_req;\0A
// CHECK-SAME:   axi_cut_a32_d64_i4_resp_t mst_resp;\0A
// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,
// CHECK-SAME:   assign mgr0_awready = slv_resp.aw_ready;\0A
// CHECK-SAME:   assign sub0_aw = '{id: mst_req.aw.id,
// CHECK-SAME:   assign mst_resp.aw_ready = sub0_awready;\0A

// A cut registers both directions, so it never bypasses
// CHECK-SAME:   axi_cut #(\0A
// CHECK-SAME:     .Bypass     (1'b0),\0A
// CHECK-SAME:   ) i_cut (\0A
// CHECK-SAME:     .slv_req_i  (slv_req),\0A
// CHECK-SAME:     .mst_resp_i (mst_resp)\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_cut_a32_d64_i4.sv">
// CHECK-SAME:  verilogName = "axi_cut_a32_d64_i4"

// CHECK-LABEL: hw.module @Cut(
// CHECK:         hw.instance "cut0" @axi_cut_a32_d64_i4(
hw.module @Cut(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %cut = axi4.cut %clk, %rst_ni, %m : !port
  hw.instance "sub" @Subordinate(axi: %cut: !port) -> ()
}

// -----

// A port with no user field neither reads nor drives PULP's, which is still a
// bit wide, so the cut side is tied off
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !port)

// Differing write and read ID widths need no PULP restriction here
// CHECK:      sv.verbatim.source
// CHECK-SAME:   typedef logic [1-1:0] axi_cut_a32_d64_i4_user_t;\0A
// CHECK-SAME:   typedef logic [4-1:0] axi_cut_a32_d64_i4_wid_t;\0A
// CHECK-SAME:   typedef logic [2-1:0] axi_cut_a32_d64_i4_rid_t;\0A
// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,
// CHECK-SAME:     region: mgr0_aw.region, atop: '0, user: '0};\0A
// CHECK-SAME:   assign mgr0_b = '{id: slv_resp.b.id, resp: slv_resp.b.resp};\0A
hw.module @NoUser(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %cut = axi4.cut %clk, %rst_ni, %m : !port
  hw.instance "sub" @Subordinate(axi: %cut: !port) -> ()
}

// -----

// With a user field both sides carry it, and PULP's atop is still tied off
!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 3, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !port)
hw.module.extern @Subordinate(in %axi : !port)

// CHECK:      sv.verbatim.source
// CHECK-SAME:   typedef logic [3-1:0] axi_cut_a32_d64_i4_user_t;\0A
// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,
// CHECK-SAME:     user: mgr0_aw.user, atop: '0};\0A
// CHECK-SAME:   assign mgr0_b = '{id: slv_resp.b.id, resp: slv_resp.b.resp, user: slv_resp.b.user};\0A
hw.module @WithUser(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !port)
  %cut = axi4.cut %clk, %rst_ni, %m : !port
  hw.instance "sub" @Subordinate(axi: %cut: !port) -> ()
}
