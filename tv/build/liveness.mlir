module {
  smt.solver() : () -> () {
    %obsF__0 = smt.declare_fun "F__0" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__1 = smt.declare_fun "F__1" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__2 = smt.declare_fun "F__2" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__3 = smt.declare_fun "F__3" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__4 = smt.declare_fun "F__4" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__5 = smt.declare_fun "F__5" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__6 = smt.declare_fun "F__6" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__7 = smt.declare_fun "F__7" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__8 = smt.declare_fun "F__8" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__9 = smt.declare_fun "F__9" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obsF__10 = smt.declare_fun "F__10" : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
    %obs0 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obsc0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %obsc0_bv32 = smt.bv.constant #smt.bv<0> : !smt.bv<32>
      %obs11 = smt.eq %obsarg1, %obsc0_bv32 : !smt.bv<32>
      %obs12 = smt.apply_func %obsF__0(%obsc0_bv16, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obs13 = smt.implies %obs11, %obs12
      smt.yield %obs13 : !smt.bool
    }
    smt.assert %obs0
    %obs1 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__0(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__1(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs1
    %obs2 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__1(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__2(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs2
    %obs3 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__2(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__3(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs3
    %obs4 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__3(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__4(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs4
    %obs5 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__4(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__5(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs5
    %obs6 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__5(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__6(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs6
    %obs7 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__6(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__7(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs7
    %obs8 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__7(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__8(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs8
    %obs9 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__8(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__9(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs9
    %obs10 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<16>, %obsarg1: !smt.bv<32>):
      %obs11 = smt.apply_func %obsF__9(%obsarg0, %obsarg1) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obsc1_bv16_0 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs12 = smt.eq %obsarg0, %obsc1_bv16_0 : !smt.bv<16>
      %obsc1_bv16_1 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs13 = smt.eq %obsc1_bv16, %obsc1_bv16_1 : !smt.bv<16>
      %obs14 = smt.bv.add %obsarg0, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv32 = smt.bv.constant #smt.bv<1> : !smt.bv<32>
      %obs15 = smt.bv.add %obsarg1, %obsc1_bv32 : !smt.bv<32>
      %obs16 = smt.apply_func %obsF__10(%obs14, %obs15) : !smt.func<(!smt.bv<16>, !smt.bv<32>) !smt.bool>
      %obstrue = smt.constant true
      %obs17 = smt.and %obs11, %obstrue
      %obs18 = smt.implies %obs17, %obs16
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs10
  }
}

