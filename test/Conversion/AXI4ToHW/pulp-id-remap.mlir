// RUN: circt-opt %s --lower-axi4-to-hw=pulp-mapping=true --split-input-file | FileCheck %s

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 6>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 6>

hw.module.extern @Manager(out axi : !wide_ids)
hw.module.extern @Subordinate(in %axi : !narrow_ids)

// CHECK:       hw.module.extern @axi_id_remap_a32_d64_i4to2_u3_usr0(
// CHECK-SAME:    attributes {source = @axi_id_remap_a32_d64_i4to2_u3_usr0.sv}
// CHECK:      sv.verbatim.source @axi_id_remap_a32_d64_i4to2_u3_usr0.sv
// CHECK-SAME:   typedef logic [4-1:0] axi_id_remap_a32_d64_i4to2_u3_usr0_slv_id_t;\0A
// CHECK-SAME:   typedef logic [2-1:0] axi_id_remap_a32_d64_i4to2_u3_usr0_mst_id_t;\0A
// CHECK-SAME:   module axi_id_remap_a32_d64_i4to2_u3_usr0 (\0A

// The table holds the IDs the remapper tracks, each taking the requests per ID
// it is given
// CHECK-SAME:   axi_id_remap #(\0A
// CHECK-SAME:     .AxiSlvPortIdWidth    (4),\0A
// CHECK-SAME:     .AxiSlvPortMaxUniqIds (3),\0A
// CHECK-SAME:     .AxiMaxTxnsPerId      (6),\0A
// CHECK-SAME:     .AxiMstPortIdWidth    (2),\0A
// CHECK-SAME:     .slv_req_t            (axi_id_remap_a32_d64_i4to2_u3_usr0_slv_req_t),\0A
// CHECK-SAME:   ) i_id_remap (\0A
// CHECK-SAME:     .slv_req_i  (slv_req[0]),\0A
// CHECK-SAME:     .mst_resp_i (mst_resp[0])\0A
// CHECK-SAME:  output_file = #hw.output_file<"axi_id_remap_a32_d64_i4to2_u3_usr0.sv">

// CHECK-LABEL: hw.module @Remapping(
// CHECK:         hw.instance "id_remap0" @axi_id_remap_a32_d64_i4to2_u3_usr0(
hw.module @Remapping(in %clk : !seq.clock, in %rst_ni : i1) {
  %m = hw.instance "mgr" @Manager() -> (axi: !wide_ids)
  %remap = axi4.id_remap %clk, %rst_ni, %m max_unique_ids = 3, concurrent_per_id = 6 : (!wide_ids) -> !narrow_ids
  hw.instance "sub" @Subordinate(axi: %remap: !narrow_ids) -> ()
}

// -----

!wide_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 4, read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>
!narrow_ids = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 2, read_id_width = 2, user_width = 0, windows = <<base = 0x0, last = 0xfff, burst_specs = <<incr, len = 4>>>>, concurrent_writes_per_id = 4, concurrent_reads_per_id = 4>

// A remapper atomics reach carries atop on both faces
// CHECK-LABEL: hw.module.extern @axi_id_remap_a32_d64_i4to2_u4_usr0_atop(
// CHECK-SAME:    in %mgr0_aw_atop : i6,
// CHECK-SAME:    out sub0_aw_atop : i6
// CHECK:      sv.verbatim.source @axi_id_remap_a32_d64_i4to2_u4_usr0_atop.sv
// CHECK-SAME:   assign sub0_aw_atop = mst_req[0].aw.atop;\0A
hw.module @Atops(in %clk : !seq.clock, in %rst_ni : i1, in %core : !wide_ids {pulp.atops}, out mem : !narrow_ids {pulp.atops}) {
  %remap = axi4.id_remap %clk, %rst_ni, %core max_unique_ids = 4, concurrent_per_id = 4 : (!wide_ids) -> !narrow_ids
  hw.output %remap : !narrow_ids
}
