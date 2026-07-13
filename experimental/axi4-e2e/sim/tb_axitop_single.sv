// Waveform testbench for designs/single.mlir. Drives AXITop's clock, dumps a
// VCD, and self-checks that the 4-beat AXI4 INCR burst read completes with
// the expected ROM words. Counterpart of sim/tb_axitop_multi.sv (2 managers);
// see run.sh Tier 3.
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
      $display("FAIL: done asserted but beats=%0h,%0h,%0h,%0h != expected %0h,%0h,%0h,%0h",
                dut.mgr_module_0.beat0, dut.mgr_module_0.beat1,
                dut.mgr_module_0.beat2, dut.mgr_module_0.beat3,
                EXPECTED_BEAT0, EXPECTED_BEAT1, EXPECTED_BEAT2, EXPECTED_BEAT3);
      $fatal;
    end else begin
      $display("PASS: burst read completed, beats=%0h,%0h,%0h,%0h",
                dut.mgr_module_0.beat0, dut.mgr_module_0.beat1,
                dut.mgr_module_0.beat2, dut.mgr_module_0.beat3);
      $finish;
    end
  end
endmodule
