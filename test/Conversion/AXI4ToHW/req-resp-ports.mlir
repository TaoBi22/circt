// RUN: circt-opt %s --lower-axi4-to-hw=req-resp-ports=true | FileCheck %s --implicit-check-not=axi4. --implicit-check-not=seq.const_clock

// Implicit check-nots ensure we drop all axi4 types & ops and all filler clocks

!port = !axi4.port<addr_width = 32, data_width = 64, write_id_width = 5, read_id_width = 3, user_width = 4, windows = <<base = 0x0, last = 0xfff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 4, outstanding_reads = 4>
!narrow = !axi4.port<addr_width = 16, data_width = 8, write_id_width = 1, read_id_width = 1, user_width = 0, windows = <<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>, outstanding_writes = 2, outstanding_reads = 2>

// The external modules the components become are internal to the lowering, so
// they keep a signal per channel
// CHECK-LABEL: hw.module.extern @axi_cut_a32_d64_i5(
// CHECK-SAME:    in %clk_i : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %mgr0_aw : !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    in %mgr0_awvalid : i1,
// CHECK-SAME:    out sub0_rready : i1)

// An output port makes the module a manager, so it drives the request and
// receives the response. The request carries PULP's `atop`, which the dialect
// does not model.
// CHECK-LABEL: hw.module.extern @ExternManager(
// CHECK-SAME:    in %clk : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %axi_resp : !hw.struct<aw_ready: i1, ar_ready: i1, w_ready: i1, b_valid: i1, b: !hw.struct<id: i5, resp: i2, user: i4>, r_valid: i1, r: !hw.struct<id: i3, data: i64, resp: i2, last: i1, user: i4>>,
// CHECK-SAME:    out axi_req : !hw.struct<aw: !hw.struct<id: i5, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i4>, aw_valid: i1, w: !hw.struct<data: i64, strb: i8, last: i1, user: i4>, w_valid: i1, b_ready: i1, ar: !hw.struct<id: i3, addr: i32, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, user: i4>, ar_valid: i1, r_ready: i1>)
hw.module.extern @ExternManager(in %clk : !seq.clock, in %rst_ni : i1, out axi : !port)

// An input port is the exact mirror: the module takes the request and drives
// the response.
// CHECK-LABEL: hw.module.extern @ExternSubordinate(
// CHECK-SAME:    in %clk : !seq.clock, in %rst_ni : i1,
// CHECK-SAME:    in %axi_req : !hw.struct<aw: !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    out axi_resp : !hw.struct<aw_ready: i1, ar_ready: i1,
hw.module.extern @ExternSubordinate(in %clk : !seq.clock, in %rst_ni : i1, in %axi : !port)

// Make sure orders are preserved when multiple ports (in both directions) are
// present. A port with no user field still gets one bit of it, since PULP's
// `user_t` is never zero width.
// CHECK-LABEL: hw.module.extern @MultiPort(
// CHECK-SAME:    in %up_req : !hw.struct<aw: !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    in %narrow_up_req : !hw.struct<aw: !hw.struct<id: i1, addr: i16, len: i8, size: i3, burst: i2, lock: i1, cache: i4, prot: i3, qos: i4, region: i4, atop: i6, user: i1>, aw_valid: i1, w: !hw.struct<data: i8, strb: i1, last: i1, user: i1>,
// CHECK-SAME:    in %down_resp : !hw.struct<aw_ready: i1,
// CHECK-SAME:    in %narrow_down_resp : !hw.struct<aw_ready: i1,
// CHECK-SAME:    out up_resp : !hw.struct<aw_ready: i1,
// CHECK-SAME:    out narrow_up_resp : !hw.struct<aw_ready: i1, ar_ready: i1, w_ready: i1, b_valid: i1, b: !hw.struct<id: i1, resp: i2, user: i1>,
// CHECK-SAME:    out down_req : !hw.struct<aw: !hw.struct<id: i5, addr: i32,
// CHECK-SAME:    out narrow_down_req : !hw.struct<aw: !hw.struct<id: i1, addr: i16,
hw.module.extern @MultiPort(in %up : !port, in %narrow_up : !narrow, out down : !port, out narrow_down : !narrow)

// A module that passes an !axi4.port through repacks it: the incoming request
// loses the fields the dialect does not carry, and the outgoing one has them
// zeroed.
// CHECK-LABEL: hw.module @Passthrough(
// CHECK:         %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready = hw.struct_explode %p_req
// CHECK:         %id, %addr, %len, %size, %burst, %lock, %cache, %prot, %qos, %region, %atop, %user = hw.struct_explode %aw
// CHECK:         %[[ZERO_USER:.+]] = hw.constant 0 : i0
// CHECK:         hw.struct_create (%id, %addr, %len, %size, %burst, %lock, %cache, %prot, %qos, %region, %[[ZERO_USER]])
// CHECK:         %[[ATOP:.+]] = hw.constant 0 : i6
// CHECK:         %[[FALSE:.+]] = hw.constant false
// CHECK:         hw.struct_create (%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %[[ATOP]], %[[FALSE]])
hw.module @Passthrough(in %p : !narrow, out q : !narrow) {
  hw.output %p : !narrow
}

// Two instances joined by a cut. The endpoints exchange requests and responses
// with the cut's signals, which stay as they are.
// CHECK-LABEL: hw.module @PointToPoint(
// CHECK:         %[[RESP:.+]] = hw.struct_create (%cut0.mgr0_awready, %cut0.mgr0_arready, %cut0.mgr0_wready, %cut0.mgr0_bvalid, %cut0.mgr0_b, %cut0.mgr0_rvalid, %cut0.mgr0_r)
// CHECK:         %mgr.axi_req = hw.instance "mgr" @ExternManager(
// CHECK-SAME:      axi_resp: %[[RESP]]:
// CHECK:         hw.instance "cut0" @axi_cut_a32_d64_i5(
// CHECK-SAME:      mgr0_aw: %{{.+}}: !hw.struct<id: i5, addr: i32,
// CHECK:         %[[REQ:.+]] = hw.struct_create (%{{.+}}, %cut0.sub0_awvalid, %cut0.sub0_w, %cut0.sub0_wvalid, %cut0.sub0_bready, %cut0.sub0_ar, %cut0.sub0_arvalid, %cut0.sub0_rready)
// CHECK:         hw.instance "sub" @ExternSubordinate(
// CHECK-SAME:      axi_req: %[[REQ]]:
hw.module @PointToPoint(in %clk : !seq.clock, in %rst_ni : i1) {
  %axi = hw.instance "mgr" @ExternManager(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1) -> (axi: !port)
  %cut = axi4.cut %clk, %rst_ni, %axi : !port
  hw.instance "sub" @ExternSubordinate(clk: %clk: !seq.clock, rst_ni: %rst_ni: i1, axi: %cut: !port) -> ()
}
