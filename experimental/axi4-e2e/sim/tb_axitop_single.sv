// Waveform testbench for designs/single.mlir. Drives AXITop's clock, dumps a
// VCD, and self-checks the full read-write-read sequence: a 4-beat read
// burst, a 2-beat write burst overwriting words 1 and 2, then a second
// 4-beat read burst confirming only those 2 words changed. Counterpart of
// sim/tb_axitop_multi.sv (2 managers); see run.sh Tier 3.
module tb_axitop_single;
  logic clk_i = 0;
  always #5 clk_i = ~clk_i;

  // NOTE: relies on `mgr_module_0` being the manager instance's lowered name
  // (AXI4ToHW.cpp names axi4.node instances `<module>_<counter>` in
  // node-processing order). Reordering axi4.node ops in single.mlir would
  // shift this and this path would stop resolving.
  AXITop dut (.clk_i(clk_i));

  localparam int CYCLE_BOUND = 50;
  localparam logic [63:0] EXPECTED_BEAT0 = 64'hCAFEF00DCAFEF00D;
  localparam logic [63:0] EXPECTED_BEAT1 = 64'hDEADBEEFDEADBEEF;
  localparam logic [63:0] EXPECTED_BEAT2 = 64'hFACEFEEDFACEFEED;
  localparam logic [63:0] EXPECTED_BEAT3 = 64'h8BADF00D8BADF00D;
  localparam logic [63:0] NEW_WORD1 = 64'hAAAAAAAA11111111;
  localparam logic [63:0] NEW_WORD2 = 64'hBBBBBBBB22222222;
  bit seen_done = 0;

  initial begin
    $dumpfile("tb_axitop_single.vcd");
    $dumpvars(0, tb_axitop_single);

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
    end else if (dut.mgr_module_0.beat0 !== EXPECTED_BEAT0 ||
                 dut.mgr_module_0.beat1 !== EXPECTED_BEAT1 ||
                 dut.mgr_module_0.beat2 !== EXPECTED_BEAT2 ||
                 dut.mgr_module_0.beat3 !== EXPECTED_BEAT3) begin
      $display("FAIL: first read burst mismatch: beats=%0h,%0h,%0h,%0h != expected %0h,%0h,%0h,%0h",
                dut.mgr_module_0.beat0, dut.mgr_module_0.beat1,
                dut.mgr_module_0.beat2, dut.mgr_module_0.beat3,
                EXPECTED_BEAT0, EXPECTED_BEAT1, EXPECTED_BEAT2, EXPECTED_BEAT3);
      $fatal;
    end else if (dut.mgr_module_0.after_beat0 !== EXPECTED_BEAT0 ||
                 dut.mgr_module_0.after_beat1 !== NEW_WORD1 ||
                 dut.mgr_module_0.after_beat2 !== NEW_WORD2 ||
                 dut.mgr_module_0.after_beat3 !== EXPECTED_BEAT3) begin
      $display("FAIL: read-after-write mismatch: after_beats=%0h,%0h,%0h,%0h != expected %0h,%0h,%0h,%0h",
                dut.mgr_module_0.after_beat0, dut.mgr_module_0.after_beat1,
                dut.mgr_module_0.after_beat2, dut.mgr_module_0.after_beat3,
                EXPECTED_BEAT0, NEW_WORD1, NEW_WORD2, EXPECTED_BEAT3);
      $fatal;
    end else begin
      $display("PASS: read-write-read sequence completed correctly");
      $display("  before write: %0h,%0h,%0h,%0h",
                dut.mgr_module_0.beat0, dut.mgr_module_0.beat1,
                dut.mgr_module_0.beat2, dut.mgr_module_0.beat3);
      $display("  after write:  %0h,%0h,%0h,%0h",
                dut.mgr_module_0.after_beat0, dut.mgr_module_0.after_beat1,
                dut.mgr_module_0.after_beat2, dut.mgr_module_0.after_beat3);
      $finish;
    end
  end
endmodule
