// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 6>
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !wide)
hw.module.extern @Subordinate(in %axi : !thin)

// The external module is unchanged by the option - the wrapper is verilog
// hanging off it, in its own file
// CHECK:       hw.module.extern @axi_dw_converter_a32_d64to32_i4(
// CHECK-SAME:    out sub0_rready : i1)
// CHECK-SAME:    attributes {source = @axi_dw_converter_a32_d64to32_i4.sv}
// CHECK:      sv.verbatim.source @axi_dw_converter_a32_d64to32_i4.sv

// The address, ID and user widths are shared, and only the data and strobe
// widths differ between the sides
// CHECK-SAME:   typedef logic [4-1:0] axi_dw_converter_a32_d64to32_i4_id_t;\0A
// CHECK-SAME:   typedef logic [64-1:0] axi_dw_converter_a32_d64to32_i4_slv_data_t;\0A
// CHECK-SAME:   typedef logic [64/8-1:0] axi_dw_converter_a32_d64to32_i4_slv_strb_t;\0A
// CHECK-SAME:   typedef logic [32-1:0] axi_dw_converter_a32_d64to32_i4_mst_data_t;\0A
// CHECK-SAME:   typedef logic [32/8-1:0] axi_dw_converter_a32_d64to32_i4_mst_strb_t;\0A

// AW, AR and B carry no data, so both sides share them, while W and R and the
// req/resp structs come in a pair
// CHECK-SAME:   `AXI_TYPEDEF_AW_CHAN_T(axi_dw_converter_a32_d64to32_i4_aw_chan_t,
// CHECK-SAME:   `AXI_TYPEDEF_W_CHAN_T(axi_dw_converter_a32_d64to32_i4_slv_w_chan_t, axi_dw_converter_a32_d64to32_i4_slv_data_t,
// CHECK-SAME:   `AXI_TYPEDEF_W_CHAN_T(axi_dw_converter_a32_d64to32_i4_mst_w_chan_t, axi_dw_converter_a32_d64to32_i4_mst_data_t,

// Each face's payload structs carry the data width of its own side
// CHECK-SAME:   axi_dw_converter_a32_d64to32_i4_slv_data_t data; axi_dw_converter_a32_d64to32_i4_slv_strb_t strb; logic last; } axi_dw_converter_a32_d64to32_i4_mgr_w_t;\0A
// CHECK-SAME:   axi_dw_converter_a32_d64to32_i4_mst_data_t data; axi_dw_converter_a32_d64to32_i4_mst_strb_t strb; logic last; } axi_dw_converter_a32_d64to32_i4_sub_w_t;\0A

// One req/resp pair per side, bridged to the ports of that side
// CHECK-SAME:   axi_dw_converter_a32_d64to32_i4_slv_req_t  slv_req;\0A
// CHECK-SAME:   axi_dw_converter_a32_d64to32_i4_mst_resp_t mst_resp;\0A
// CHECK-SAME:   assign slv_req.aw = '{id: mgr0_aw.id,
// CHECK-SAME:   assign sub0_aw = '{id: mst_req.aw.id,

// CHECK-SAME:   axi_dw_converter #(\0A
// The tracker count comes from the reads the upstream port can have
// outstanding
// CHECK-SAME:     .AxiMaxReads         (6),\0A
// CHECK-SAME:     .AxiSlvPortDataWidth (64),\0A
// CHECK-SAME:     .AxiMstPortDataWidth (32),\0A
// CHECK-SAME:     .AxiAddrWidth        (32),\0A
// CHECK-SAME:     .AxiIdWidth          (4),\0A
// CHECK-SAME:   ) i_dw_converter (\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_dw_converter_a32_d64to32_i4.sv">

// CHECK-LABEL: hw.module @Narrowing(
// CHECK:         hw.instance "dw_converter0" @axi_dw_converter_a32_d64to32_i4(
hw.module @Narrowing(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !wide)
  %dwc = axi4.data_width_converter %clk, %rst_ni, %m : (!wide) -> !thin
  hw.instance "sub" @Subordinate(axi: %dwc: !thin) -> ()
}

// -----

// Widening is the same wrapper with the two data widths the other way round
!thin = !axi4.port<addr_width = 32, data_width = 32, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 8>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !thin)
hw.module.extern @Subordinate(in %axi : !wide)

// CHECK:      sv.verbatim.source @axi_dw_converter_a32_d32to64_i4.sv
// CHECK-SAME:   typedef logic [32-1:0] axi_dw_converter_a32_d32to64_i4_slv_data_t;\0A
// CHECK-SAME:   typedef logic [64-1:0] axi_dw_converter_a32_d32to64_i4_mst_data_t;\0A
// CHECK-SAME:     .AxiSlvPortDataWidth (32),\0A
// CHECK-SAME:     .AxiMstPortDataWidth (64),\0A
hw.module @Widening(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !thin)
  %dwc = axi4.data_width_converter %clk, %rst_ni, %m : (!thin) -> !wide
  hw.instance "sub" @Subordinate(axi: %dwc: !wide) -> ()
}
