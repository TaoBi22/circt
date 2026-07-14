// Waveform testbench for designs/mixed_fanout.mlir. Drives AXITop's clock,
// dumps a VCD, and self-checks that TWO SEQUENTIAL 7-phase read-write-read
// sequences (one manager, one direct via xbar1 to sub_module_a, one chained
// through xbar2 to sub_module_b) each complete correctly -- a 4-beat read
// burst, a 2-beat write burst overwriting words 1 and 2, then a second
// 4-beat read burst confirming only those 2 words changed -- proving both
// fan-out paths of the real PULP axi_xbar support read-after-write
// consistency, not just single-flow correctness. Counterpart of
// sim/tb_axitop_single.sv / sim/tb_axitop_multi.sv (see run.sh Tier 3).
module tb_axitop_mixed_fanout;
  logic clk_i = 0;
  always #5 clk_i = ~clk_i;

  // NOTE: relies on `mgr_module_0` being the manager instance's lowered name
  // (AXI4ToHW.cpp names axi4.node instances `<module>_<counter>`, counter
  // shared across all managers/subordinates/xbars, assigned in node-
  // processing order -- see NetworkLowering's instanceCounter in
  // lib/Conversion/AXI4ToHW/AXI4ToHW.cpp). mixed_fanout.mlir's textual
  // axi4.node order is mnode(mgr_module), snode_a(sub_module_a),
  // snode_b(sub_module_b), which is what makes this path resolve to _0.
  // Reordering the axi4.node ops shifts every name after that point and
  // this path would stop resolving.
  AXITop dut (.clk_i(clk_i));

  localparam int CYCLE_BOUND = 120;
  localparam logic [63:0] EXPECTED_A_BEAT0 = 64'hCAFEF00DCAFEF00D;
  localparam logic [63:0] EXPECTED_A_BEAT1 = 64'hDEADBEEFDEADBEEF;
  localparam logic [63:0] EXPECTED_A_BEAT2 = 64'hFACEFEEDFACEFEED;
  localparam logic [63:0] EXPECTED_A_BEAT3 = 64'h8BADF00D8BADF00D;
  localparam logic [63:0] NEW_A_WORD1 = 64'hAAAAAAAA11111111;
  localparam logic [63:0] NEW_A_WORD2 = 64'hBBBBBBBB22222222;
  localparam logic [63:0] EXPECTED_B_BEAT0 = 64'hFEEDFACEFEEDFACE;
  localparam logic [63:0] EXPECTED_B_BEAT1 = 64'hB105F00DB105F00D;
  localparam logic [63:0] EXPECTED_B_BEAT2 = 64'h5CA1AB1E5CA1AB1E;
  localparam logic [63:0] EXPECTED_B_BEAT3 = 64'h0DEFACED0DEFACED;
  localparam logic [63:0] NEW_B_WORD1 = 64'hCCCCCCCC33333333;
  localparam logic [63:0] NEW_B_WORD2 = 64'hDDDDDDDD44444444;
  bit seen_done = 0;

  initial begin
    $dumpfile("tb_axitop_mixed_fanout.vcd");
    $dumpvars(0, tb_axitop_mixed_fanout);

    for (int i = 0; i < CYCLE_BOUND; i++) begin
      @(posedge clk_i);
      if (dut.mgr_module_0.done) begin
        seen_done = 1;
        break;
      end
    end

    if (!seen_done) begin
      $display("FAIL: timeout after %0d cycles without done", CYCLE_BOUND);
      $fatal;
    end else if (dut.mgr_module_0.a_beat0 !== EXPECTED_A_BEAT0 ||
                 dut.mgr_module_0.a_beat1 !== EXPECTED_A_BEAT1 ||
                 dut.mgr_module_0.a_beat2 !== EXPECTED_A_BEAT2 ||
                 dut.mgr_module_0.a_beat3 !== EXPECTED_A_BEAT3 ||
                 dut.mgr_module_0.b_beat0 !== EXPECTED_B_BEAT0 ||
                 dut.mgr_module_0.b_beat1 !== EXPECTED_B_BEAT1 ||
                 dut.mgr_module_0.b_beat2 !== EXPECTED_B_BEAT2 ||
                 dut.mgr_module_0.b_beat3 !== EXPECTED_B_BEAT3) begin
      $display("FAIL: first read burst mismatch");
      $display("  burst A (direct via xbar1): %0h,%0h,%0h,%0h",
                dut.mgr_module_0.a_beat0, dut.mgr_module_0.a_beat1,
                dut.mgr_module_0.a_beat2, dut.mgr_module_0.a_beat3);
      $display("  burst B (chained via xbar2): %0h,%0h,%0h,%0h",
                dut.mgr_module_0.b_beat0, dut.mgr_module_0.b_beat1,
                dut.mgr_module_0.b_beat2, dut.mgr_module_0.b_beat3);
      $fatal;
    end else if (dut.mgr_module_0.after_a_beat0 !== EXPECTED_A_BEAT0 ||
                 dut.mgr_module_0.after_a_beat1 !== NEW_A_WORD1 ||
                 dut.mgr_module_0.after_a_beat2 !== NEW_A_WORD2 ||
                 dut.mgr_module_0.after_a_beat3 !== EXPECTED_A_BEAT3 ||
                 dut.mgr_module_0.after_b_beat0 !== EXPECTED_B_BEAT0 ||
                 dut.mgr_module_0.after_b_beat1 !== NEW_B_WORD1 ||
                 dut.mgr_module_0.after_b_beat2 !== NEW_B_WORD2 ||
                 dut.mgr_module_0.after_b_beat3 !== EXPECTED_B_BEAT3) begin
      $display("FAIL: read-after-write mismatch");
      $display("  burst A after: %0h,%0h,%0h,%0h",
                dut.mgr_module_0.after_a_beat0, dut.mgr_module_0.after_a_beat1,
                dut.mgr_module_0.after_a_beat2, dut.mgr_module_0.after_a_beat3);
      $display("  burst B after: %0h,%0h,%0h,%0h",
                dut.mgr_module_0.after_b_beat0, dut.mgr_module_0.after_b_beat1,
                dut.mgr_module_0.after_b_beat2, dut.mgr_module_0.after_b_beat3);
      $fatal;
    end else begin
      $display("PASS: both fan-out paths' read-write-read sequences completed correctly");
      $display("  sub_module_a (direct via xbar1)");
      $display("    before write: %0h,%0h,%0h,%0h",
                dut.mgr_module_0.a_beat0, dut.mgr_module_0.a_beat1,
                dut.mgr_module_0.a_beat2, dut.mgr_module_0.a_beat3);
      $display("    after write:  %0h,%0h,%0h,%0h",
                dut.mgr_module_0.after_a_beat0, dut.mgr_module_0.after_a_beat1,
                dut.mgr_module_0.after_a_beat2, dut.mgr_module_0.after_a_beat3);
      $display("  sub_module_b (chained via xbar2)");
      $display("    before write: %0h,%0h,%0h,%0h",
                dut.mgr_module_0.b_beat0, dut.mgr_module_0.b_beat1,
                dut.mgr_module_0.b_beat2, dut.mgr_module_0.b_beat3);
      $display("    after write:  %0h,%0h,%0h,%0h",
                dut.mgr_module_0.after_b_beat0, dut.mgr_module_0.after_b_beat1,
                dut.mgr_module_0.after_b_beat2, dut.mgr_module_0.after_b_beat3);
      $finish;
    end
  end
endmodule
