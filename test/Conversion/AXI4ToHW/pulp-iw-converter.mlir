// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 6>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !wide_ids)
hw.module.extern @Subordinate(in %axi : !narrow_ids)

// The external module is unchanged by the option - the wrapper is verilog
// hanging off it, in its own file
// CHECK:       hw.module.extern @axi_iw_converter_a32_d64_i4to2(
// CHECK-SAME:    out sub0_rready : i1)
// CHECK-SAME:    attributes {source = @axi_iw_converter_a32_d64_i4to2.sv}
// CHECK:      sv.verbatim.source @axi_iw_converter_a32_d64_i4to2.sv

// Only the ID widths differ between the sides, so each side gets an ID typedef
// and a channel type family of its own
// CHECK-SAME:   typedef logic [4-1:0] axi_iw_converter_a32_d64_i4to2_slv_id_t;\0A
// CHECK-SAME:   typedef logic [2-1:0] axi_iw_converter_a32_d64_i4to2_mst_id_t;\0A
// CHECK-SAME:   `AXI_TYPEDEF_ALL(axi_iw_converter_a32_d64_i4to2_slv, axi_iw_converter_a32_d64_i4to2_addr_t, axi_iw_converter_a32_d64_i4to2_slv_id_t,
// CHECK-SAME:   `AXI_TYPEDEF_ALL(axi_iw_converter_a32_d64_i4to2_mst, axi_iw_converter_a32_d64_i4to2_addr_t, axi_iw_converter_a32_d64_i4to2_mst_id_t,

// Each face's payload structs carry the ID width of its own side
// CHECK-SAME:   typedef struct packed { axi_iw_converter_a32_d64_i4to2_slv_id_t id; {{.*}} } axi_iw_converter_a32_d64_i4to2_mgr_aw_t;\0A
// CHECK-SAME:   typedef struct packed { axi_iw_converter_a32_d64_i4to2_mst_id_t id; {{.*}} } axi_iw_converter_a32_d64_i4to2_sub_aw_t;\0A

// One face per side, bridged to the single struct of the array
// CHECK-SAME:   module axi_iw_converter_a32_d64_i4to2 (\0A
// CHECK-SAME:   axi_iw_converter_a32_d64_i4to2_slv_req_t  [1-1:0] slv_req;\0A
// CHECK-SAME:   assign slv_req[0].aw = '{id: mgr0_aw.id,
// CHECK-SAME:   assign sub0_aw = '{id: mst_req[0].aw.id,

// PULP sizes its tables from the transactions each side keeps in flight. The 6
// unique upstream IDs outnumber the 4 the downstream side has, so PULP
// serialises rather than remaps them.
// CHECK-SAME:   axi_iw_converter #(\0A
// CHECK-SAME:     .AxiSlvPortIdWidth      (4),\0A
// CHECK-SAME:     .AxiMstPortIdWidth      (2),\0A
// CHECK-SAME:     .AxiSlvPortMaxUniqIds   (6),\0A
// CHECK-SAME:     .AxiSlvPortMaxTxnsPerId (6),\0A
// CHECK-SAME:     .AxiSlvPortMaxTxns      (6),\0A
// CHECK-SAME:     .AxiMstPortMaxUniqIds   (4),\0A
// CHECK-SAME:     .AxiMstPortMaxTxnsPerId (4),\0A
// CHECK-SAME:     .AxiAddrWidth           (32),\0A
// CHECK-SAME:     .AxiDataWidth           (64),\0A
// CHECK-SAME:     .AxiUserWidth           (1),\0A
// CHECK-SAME:   ) i_iw_converter (\0A
// CHECK-SAME:     .slv_req_i  (slv_req[0]),\0A
// CHECK-SAME:     .mst_resp_i (mst_resp[0])\0A

// CHECK-SAME:  output_file = #hw.output_file<"axi_iw_converter_a32_d64_i4to2.sv">

// CHECK-LABEL: hw.module @Narrowing(
// CHECK:         hw.instance "iw_converter0" @axi_iw_converter_a32_d64_i4to2(
hw.module @Narrowing(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !wide_ids)
  %iwc = axi4.id_width_converter %clk, %rst_ni, %m : (!wide_ids) -> !narrow_ids
  hw.instance "sub" @Subordinate(axi: %iwc: !narrow_ids) -> ()
}

// -----

// Widening is the same wrapper with the two ID widths the other way round
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>

hw.module.extern @Manager(out axi : !narrow_ids)
hw.module.extern @Subordinate(in %axi : !wide_ids)

// CHECK:      sv.verbatim.source @axi_iw_converter_a32_d64_i2to4.sv
// CHECK-SAME:   typedef logic [2-1:0] axi_iw_converter_a32_d64_i2to4_slv_id_t;\0A
// CHECK-SAME:   typedef logic [4-1:0] axi_iw_converter_a32_d64_i2to4_mst_id_t;\0A
// CHECK-SAME:     .AxiSlvPortIdWidth      (2),\0A
// CHECK-SAME:     .AxiMstPortIdWidth      (4),\0A
hw.module @Widening(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !narrow_ids)
  %iwc = axi4.id_width_converter %clk, %rst_ni, %m : (!narrow_ids) -> !wide_ids
  hw.instance "sub" @Subordinate(axi: %iwc: !wide_ids) -> ()
}
