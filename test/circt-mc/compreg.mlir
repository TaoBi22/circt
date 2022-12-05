// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s
// RUN: circt-opt %s -verify-diagnostics --lower-seq-to-sv | circt-opt -verify-diagnostics | FileCheck %s --check-prefix=SV
hw.module @top(%clk: i1, %rst: i1, %i: i32) {
  %rv = hw.constant 0 : i32

  %r0 = seq.compreg %i, %clk, %rst, %rv : i32
  seq.compreg %i, %clk : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk, %rst, %c0_i32  : i32
  // CHECK: %{{.+}} = seq.compreg %i, %clk : i32
  // SV: [[REG0:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: [[REG5:%.+]] = sv.read_inout [[REG0]] : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG0]], %i : i32
  // SV: }(syncreset : posedge %rst)  {
  // SV:   sv.passign [[REG0]], %c0_i32 : i32
  // SV: }
  // SV: [[REG1:%.+]] = sv.reg  : !hw.inout<i32>
  // SV: sv.alwaysff(posedge %clk)  {
  // SV:   sv.passign [[REG1]], %i : i32
  // SV: }
}
