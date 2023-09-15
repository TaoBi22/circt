// These tests will be only enabled if circt-mc is built.
// REQUIRES: circt-mc


//  RUN: circt-mc %s -b 10 --module OrCommutes | FileCheck %s --check-prefix=ORCOMMUTES
//  ORCOMMUTES: Success!

hw.module @OrCommutes(%i0: i1, %i1: i1) {
  %or0 = comb.or bin %i0, %i1 : i1
  %or1 = comb.or bin %i1, %i0 : i1
  // Condition
  %cond = comb.icmp bin eq %or0, %or1 : i1
  verif.assert %cond : i1
}

//  RUN: circt-mc %s -b 10 --module demorgan | FileCheck %s --check-prefix=DEMORGAN
//  DEMORGAN: Success!

hw.module @demorgan(%i0: i1, %i1: i1) {
  %c1 = hw.constant 1 : i1
  %ni0 = comb.xor bin %i0, %c1 : i1
  %ni1 = comb.xor bin %i1, %c1 : i1
  %or = comb.or bin %ni0, %ni1 : i1
  // Condition
  %and = comb.and bin %i0, %i1 : i1
  %nand = comb.xor bin %and, %c1 : i1
  %cond = comb.icmp bin neq %or, %nand : i1
  verif.assert %cond : i1
}

//  RUN: circt-mc %s -b 10 --module OrderInvariance | FileCheck %s --check-prefix=ORDERINVARIANCE
//  ORDERINVARIANCE: Success!

hw.module @OrderInvariance(%i0: i1, %i1: i1, %i2: i1, %clk: i1) {
  %or0 = comb.or bin %i0, %i1 : i1
  %and0 = comb.and bin %or0, %i2 : i1
  %res0 = seq.compreg %and0, %clk : i1
  // Same but in reverse order
  %res1 = seq.compreg %and1, %clk : i1
  %and1 = comb.and bin %or1, %i2 : i1
  %or1 = comb.or bin %i0, %i1 : i1
  // Condition
  %cond = comb.icmp bin eq %res0, %res1 : i1
  verif.assert %cond : i1
}