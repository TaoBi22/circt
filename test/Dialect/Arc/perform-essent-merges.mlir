// RUN: circt-opt %s --perform-essent-merges="optimal-partition-size=3" | FileCheck %s

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
arc.define @SmallSibling1(%a: i8, %b: i8) -> i8 {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @SmallSibling2(%a: i8, %b: i8) -> i8 {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @Parent1(%a : i8) -> (i8) {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @Parent2(%a : i8) -> (i8) {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

hw.module @MergeSmallSiblings() {
  %0 = hw.constant 0 : i8
  %parent1 = arc.call @Parent1(%0) : (i8) -> (i8)
  %parent2 = arc.call @Parent2(%0) : (i8) -> (i8)
  %ssib1 = arc.call @SmallSibling1(%parent1, %parent2) : (i8, i8) -> (i8)
  %ssib2 = arc.call @SmallSibling2(%parent1, %parent2) : (i8, i8) -> (i8)
}

// Check small sibling into big sibling merges

// CHECK-LABEL: arc.define
arc.define @SmallSibling(%a: i8, %b: i8) -> i8 {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @BigSibling(%a: i8, %b: i8) -> i8 {
  %and = comb.and %a, %a : i8
  %and1 = comb.and %a, %a : i8
  %and2 = comb.and %a, %a : i8
  %and3 = comb.and %a, %a : i8
  %and4 = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @Parent3(%a : i8) -> (i8) {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

arc.define @Parent4(%a : i8) -> (i8) {
  %and = comb.and %a, %a : i8
  arc.output %and : i8
}

hw.module @MergeSmallAndBigSiblings() {
  %0 = hw.constant 0 : i8
  %parent1 = arc.call @Parent3(%0) : (i8) -> (i8)
  %parent2 = arc.call @Parent4(%0) : (i8) -> (i8)
  %ssib1 = arc.call @BigSibling(%parent1, %parent2) : (i8, i8) -> (i8)
  %ssib2 = arc.call @SmallSibling(%parent1, %parent2) : (i8, i8) -> (i8)
}
