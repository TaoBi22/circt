// Waveform testbench for designs/multi.mlir. Drives AXITop's clock, dumps a
// VCD, and self-checks that TWO CONCURRENT 4-beat AXI4 INCR burst reads (one
// per manager, through the same shared xbar, to two DIFFERENT subordinates)
// each complete with the expected, DISTINCT ROM words -- proving the real
// PULP axi_xbar's address routing, inter-manager arbitration, and downstream
// id-widening (4 -> 5 bits) actually work end to end, not just single-flow
// correctness. multi-design-only counterpart of sim/tb_axitop_single.sv (see
// run.sh Tier 3).
module tb_axitop_multi;
  logic clk_i = 0;
  always #5 clk_i = ~clk_i;

  // NOTE: relies on `mgr_module_a_0` / `mgr_module_b_1` being the two manager
  // instances' lowered names (AXI4ToHW.cpp names axi4.node instances
  // `<module>_<counter>`, counter shared across all managers/subordinates/
  // xbars, assigned in node-processing order -- see NetworkLowering's
  // instanceCounter in lib/Conversion/AXI4ToHW/AXI4ToHW.cpp). multi.mlir's
  // textual axi4.node order is mnode1(mgr_module_a), mnode2(mgr_module_b),
  // snode1(sub_module5_a), snode2(sub_module5_b), which is what makes these
  // paths resolve to _0/_1 (and the subordinates to sub_module5_a_2/
  // sub_module5_b_3). Reordering the axi4.node ops shifts every name after
  // that point and these paths would stop resolving.
  AXITop dut (.clk_i(clk_i));

  localparam int CYCLE_BOUND = 50;
  localparam logic [63:0] EXPECTED_A_BEAT0 = 64'hCAFEF00DCAFEF00D;
  localparam logic [63:0] EXPECTED_A_BEAT1 = 64'hDEADBEEFDEADBEEF;
  localparam logic [63:0] EXPECTED_A_BEAT2 = 64'hFACEFEEDFACEFEED;
  localparam logic [63:0] EXPECTED_A_BEAT3 = 64'h8BADF00D8BADF00D;
  localparam logic [63:0] EXPECTED_B_BEAT0 = 64'hFEEDFACEFEEDFACE;
  localparam logic [63:0] EXPECTED_B_BEAT1 = 64'hB105F00DB105F00D;
  localparam logic [63:0] EXPECTED_B_BEAT2 = 64'h5CA1AB1E5CA1AB1E;
  localparam logic [63:0] EXPECTED_B_BEAT3 = 64'h0DEFACED0DEFACED;

  bit seen_a = 0, seen_b = 0;
  int done_cycle_a = -1, done_cycle_b = -1;

  initial begin
    $dumpfile("tb_axitop_multi.vcd");
    $dumpvars(0, tb_axitop_multi);

    for (int i = 0; i < CYCLE_BOUND; i++) begin
      @(posedge clk_i);
      if (!seen_a && dut.mgr_module_a_0.done) begin
        seen_a = 1;
        done_cycle_a = i;
      end
      if (!seen_b && dut.mgr_module_b_1.done) begin
        seen_b = 1;
        done_cycle_b = i;
      end
      if (seen_a && seen_b) break;
    end

    if (!seen_a || !seen_b) begin
      $display("FAIL: timeout after %0d cycles (mgr_a done=%0d, mgr_b done=%0d)",
                CYCLE_BOUND, seen_a, seen_b);
      $fatal;
    end else if (dut.mgr_module_a_0.beat0 !== EXPECTED_A_BEAT0 ||
                 dut.mgr_module_a_0.beat1 !== EXPECTED_A_BEAT1 ||
                 dut.mgr_module_a_0.beat2 !== EXPECTED_A_BEAT2 ||
                 dut.mgr_module_a_0.beat3 !== EXPECTED_A_BEAT3 ||
                 dut.mgr_module_b_1.beat0 !== EXPECTED_B_BEAT0 ||
                 dut.mgr_module_b_1.beat1 !== EXPECTED_B_BEAT1 ||
                 dut.mgr_module_b_1.beat2 !== EXPECTED_B_BEAT2 ||
                 dut.mgr_module_b_1.beat3 !== EXPECTED_B_BEAT3) begin
      $display("FAIL: done asserted but beats mismatch");
      $display("  mgr_a (done@%0d): %0h,%0h,%0h,%0h", done_cycle_a,
                dut.mgr_module_a_0.beat0, dut.mgr_module_a_0.beat1,
                dut.mgr_module_a_0.beat2, dut.mgr_module_a_0.beat3);
      $display("  mgr_b (done@%0d): %0h,%0h,%0h,%0h", done_cycle_b,
                dut.mgr_module_b_1.beat0, dut.mgr_module_b_1.beat1,
                dut.mgr_module_b_1.beat2, dut.mgr_module_b_1.beat3);
      $fatal;
    end else begin
      $display("PASS: both concurrent burst reads completed through the shared xbar");
      $display("  mgr_a -> sub_module5_a (done@cycle %0d): %0h,%0h,%0h,%0h", done_cycle_a,
                dut.mgr_module_a_0.beat0, dut.mgr_module_a_0.beat1,
                dut.mgr_module_a_0.beat2, dut.mgr_module_a_0.beat3);
      $display("  mgr_b -> sub_module5_b (done@cycle %0d): %0h,%0h,%0h,%0h", done_cycle_b,
                dut.mgr_module_b_1.beat0, dut.mgr_module_b_1.beat1,
                dut.mgr_module_b_1.beat2, dut.mgr_module_b_1.beat3);
      $finish;
    end
  end
endmodule
