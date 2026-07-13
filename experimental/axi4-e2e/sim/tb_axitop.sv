// Waveform testbench for designs/single.mlir. Drives AXITop's clock, dumps a
// VCD, and self-checks that the single AXI4 read completes with the expected
// ROM word. single-design-only (see run.sh Tier 3).
module tb_axitop;
  logic clk_i = 0;
  always #5 clk_i = ~clk_i;

  // NOTE: relies on `mgr_module_0` being the manager instance's lowered name
  // (AXI4ToHW.cpp names axi4.node instances `<module>_<counter>` in
  // node-processing order). Reordering axi4.node ops in single.mlir would
  // shift this and this path would stop resolving.
  AXITop dut (.clk_i(clk_i));

  localparam int CYCLE_BOUND = 50;
  localparam logic [63:0] EXPECTED_WORD0 = 64'hCAFEF00DCAFEF00D;
  bit seen_done = 0;

  initial begin
    $dumpfile("tb_axitop.vcd");
    $dumpvars(0, tb_axitop);

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
    end else if (dut.mgr_module_0.captured_rdata !== EXPECTED_WORD0) begin
      $display("FAIL: done asserted but captured_rdata=%0h != expected %0h",
                dut.mgr_module_0.captured_rdata, EXPECTED_WORD0);
      $fatal;
    end else begin
      $display("PASS: read completed, captured_rdata=%0h", dut.mgr_module_0.captured_rdata);
      $finish;
    end
  end
endmodule
