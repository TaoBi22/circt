module {
  smt.solver() : () -> () {
    %obsF__0 = smt.declare_fun "F__0" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__1 = smt.declare_fun "F__1" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__2 = smt.declare_fun "F__2" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__3 = smt.declare_fun "F__3" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__4 = smt.declare_fun "F__4" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__5 = smt.declare_fun "F__5" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__6 = smt.declare_fun "F__6" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__7 = smt.declare_fun "F__7" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__8 = smt.declare_fun "F__8" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__9 = smt.declare_fun "F__9" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF__10 = smt.declare_fun "F__10" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obsF_ERR = smt.declare_fun "F_ERR" : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
    %obs0 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.int, %obsarg2: !smt.int):
      %obsc0 = smt.int.constant 0
      %obsc0_0 = smt.int.constant 0
      %obs21 = smt.eq %obsarg2, %obsc0_0 : !smt.int
      %obs22 = smt.apply_func %obsF__0(%obsarg0, %obsc0, %obsarg2) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obs23 = smt.implies %obs21, %obs22
      smt.yield %obs23 : !smt.bool
    }
    smt.assert %obs0
    %obs1 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__0(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__1(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs1
    %obs2 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__0(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs2
    %obs3 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__1(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__2(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs3
    %obs4 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__1(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs4
    %obs5 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__2(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__3(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs5
    %obs6 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__2(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs6
    %obs7 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__3(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__4(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs7
    %obs8 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__3(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs8
    %obs9 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__4(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__5(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs9
    %obs10 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__4(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs10
    %obs11 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__5(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__6(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs11
    %obs12 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__5(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs12
    %obs13 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__6(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__7(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs13
    %obs14 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__6(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs14
    %obs15 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__7(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__8(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs15
    %obs16 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__7(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs16
    %obs17 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__8(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__9(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs17
    %obs18 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__8(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs18
    %obs19 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__9(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg2, %obsc1
      %obsc65536 = smt.int.constant 65536
      %obs23 = smt.int.mod %obs22, %obsc65536
      %obsc1_0 = smt.int.constant 1
      %obs24 = smt.int.add %obsarg3, %obsc1_0
      %obs25 = smt.apply_func %obsF__10(%obsarg0, %obs23, %obs24) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs26 = smt.distinct %obsarg1, %obstrue : !smt.bool
      %obs27 = smt.and %obs21, %obs26
      %obs28 = smt.implies %obs27, %obs25
      smt.yield %obs28 : !smt.bool
    }
    smt.assert %obs19
    %obs20 = smt.forall {
    ^bb0(%obsarg0: !smt.bool, %obsarg1: !smt.bool, %obsarg2: !smt.int, %obsarg3: !smt.int):
      %obs21 = smt.apply_func %obsF__9(%obsarg1, %obsarg2, %obsarg3) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs22 = smt.int.add %obsarg3, %obsc1
      %obs23 = smt.apply_func %obsF_ERR(%obsarg0, %obsarg2, %obs22) : !smt.func<(!smt.bool, !smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs24 = smt.eq %obsarg1, %obstrue : !smt.bool
      %obs25 = smt.and %obs21, %obs24
      %obs26 = smt.implies %obs25, %obs23
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs20
  }
}

