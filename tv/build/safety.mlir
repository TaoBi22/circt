module {
  smt.solver() : () -> () {
    %obsF__0 = smt.declare_fun "F__0" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__1 = smt.declare_fun "F__1" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__2 = smt.declare_fun "F__2" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__3 = smt.declare_fun "F__3" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__4 = smt.declare_fun "F__4" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__5 = smt.declare_fun "F__5" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__6 = smt.declare_fun "F__6" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__7 = smt.declare_fun "F__7" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__8 = smt.declare_fun "F__8" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__9 = smt.declare_fun "F__9" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__10 = smt.declare_fun "F__10" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obs0 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obsc0 = smt.int.constant 0
      %obs11 = smt.apply_func %obsF__0(%obsc0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc0_0 = smt.int.constant 0
      %obs12 = smt.eq %obsarg1, %obsc0_0 : !smt.int
      %obs13 = smt.implies %obs12, %obs11
      smt.yield %obs13 : !smt.bool
    }
    smt.assert %obs0
    %obs1 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__0(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__1(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs1
    %obs2 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__1(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__2(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs2
    %obs3 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__2(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__3(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs3
    %obs4 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__3(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__4(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs4
    %obs5 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__4(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__5(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs5
    %obs6 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__5(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__6(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs6
    %obs7 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__6(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__7(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs7
    %obs8 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__7(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__8(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs8
    %obs9 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__8(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__9(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs9
    %obs10 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs11 = smt.apply_func %obsF__9(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs12 = smt.int.add %obsarg0, %obsc1
      %obsc1_0 = smt.int.constant 1
      %obs13 = smt.int.add %obsarg1, %obsc1_0
      %obs14 = smt.apply_func %obsF__10(%obs12, %obs13) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs15 = smt.and %obs11, %obstrue
      %obs16 = smt.implies %obs15, %obs14
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs10
  }
}

