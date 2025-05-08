// RUN: circt-opt %s --perform-essent-merges="optimal-partition-size=2" | FileCheck %s

// Check single parent merges

// CHECK-LABEL: arc.define
arc.define @EmptyArc1(%a: i8) -> i8 {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @EmptyArc2(%a : i7, %c : i7, %b : i8) -> (i7, i8) {
  %1 = comb.xor %a, %c : i7
  %2 = comb.or %b : i8
  arc.output %1, %2 : i7, i8
}

hw.module @callInModuleTest() {
  %0 = hw.constant 0 : i8
  %1 = hw.constant 0 : i7
  arc.call @EmptyArc1(%res#1) : (i8) -> (i8)
  %res:2 = arc.call @EmptyArc2(%1, %1, %0) : (i7, i7, i8) -> (i7, i8)
}

// Check small sibling merges

// CHECK-LABEL: arc.define
arc.define @SmallSibling1(%a: i8) -> i8 {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @SmallSibling2(%a: i8) -> i8 {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @Parent(%a : i8) -> (i8) {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

hw.module @MergeSmallSiblings() {
  %0 = hw.constant 0 : i8
  %parent = arc.call @Parent(%0) : (i8) -> (i8)
  %ssib1 = arc.call @SmallSibling1(%parent) : (i8) -> (i8)
  %ssib2 = arc.call @SmallSibling2(%0) : (i8) -> (i8)
}
