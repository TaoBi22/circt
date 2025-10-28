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
    %obsF__11 = smt.declare_fun "F__11" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__12 = smt.declare_fun "F__12" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__13 = smt.declare_fun "F__13" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__14 = smt.declare_fun "F__14" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__15 = smt.declare_fun "F__15" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__16 = smt.declare_fun "F__16" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__17 = smt.declare_fun "F__17" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__18 = smt.declare_fun "F__18" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__19 = smt.declare_fun "F__19" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__20 = smt.declare_fun "F__20" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__21 = smt.declare_fun "F__21" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__22 = smt.declare_fun "F__22" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__23 = smt.declare_fun "F__23" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__24 = smt.declare_fun "F__24" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__25 = smt.declare_fun "F__25" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__26 = smt.declare_fun "F__26" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__27 = smt.declare_fun "F__27" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__28 = smt.declare_fun "F__28" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__29 = smt.declare_fun "F__29" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__30 = smt.declare_fun "F__30" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__31 = smt.declare_fun "F__31" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__32 = smt.declare_fun "F__32" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__33 = smt.declare_fun "F__33" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__34 = smt.declare_fun "F__34" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__35 = smt.declare_fun "F__35" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__36 = smt.declare_fun "F__36" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__37 = smt.declare_fun "F__37" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__38 = smt.declare_fun "F__38" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__39 = smt.declare_fun "F__39" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__40 = smt.declare_fun "F__40" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__41 = smt.declare_fun "F__41" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__42 = smt.declare_fun "F__42" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__43 = smt.declare_fun "F__43" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__44 = smt.declare_fun "F__44" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__45 = smt.declare_fun "F__45" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__46 = smt.declare_fun "F__46" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__47 = smt.declare_fun "F__47" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__48 = smt.declare_fun "F__48" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__49 = smt.declare_fun "F__49" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obsF__50 = smt.declare_fun "F__50" : !smt.func<(!smt.int, !smt.int) !smt.bool>
    %obs0 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obsc0 = smt.int.constant 0
      %obs51 = smt.apply_func %obsF__0(%obsc0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc0_0 = smt.int.constant 0
      %obs52 = smt.eq %obsarg1, %obsc0_0 : !smt.int
      %obs53 = smt.implies %obs52, %obs51
      smt.yield %obs53 : !smt.bool
    }
    smt.assert %obs0
    %obs1 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__0(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__1(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs1
    %obs2 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__1(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__2(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs2
    %obs3 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__2(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__3(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs3
    %obs4 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__3(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__4(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs4
    %obs5 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__4(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__5(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs5
    %obs6 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__5(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__6(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs6
    %obs7 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__6(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__7(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs7
    %obs8 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__7(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__8(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs8
    %obs9 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__8(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__9(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs9
    %obs10 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__9(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__10(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs10
    %obs11 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__10(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__11(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs11
    %obs12 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__11(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__12(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs12
    %obs13 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__12(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__13(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs13
    %obs14 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__13(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__14(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs14
    %obs15 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__14(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__15(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs15
    %obs16 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__15(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__16(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs16
    %obs17 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__16(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__17(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs17
    %obs18 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__17(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__18(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs18
    %obs19 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__18(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__19(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs19
    %obs20 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__19(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__20(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs20
    %obs21 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__20(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__21(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs21
    %obs22 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__21(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__22(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs22
    %obs23 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__22(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__23(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs23
    %obs24 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__23(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__24(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs24
    %obs25 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__24(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__25(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs25
    %obs26 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__25(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__26(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs26
    %obs27 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__26(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__27(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs27
    %obs28 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__27(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__28(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs28
    %obs29 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__28(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__29(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs29
    %obs30 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__29(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__30(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs30
    %obs31 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__30(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__31(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs31
    %obs32 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__31(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__32(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs32
    %obs33 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__32(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__33(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs33
    %obs34 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__33(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__34(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs34
    %obs35 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__34(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__35(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs35
    %obs36 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__35(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__36(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs36
    %obs37 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__36(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__37(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs37
    %obs38 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__37(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__38(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs38
    %obs39 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__38(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__39(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs39
    %obs40 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__39(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__40(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs40
    %obs41 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__40(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__41(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs41
    %obs42 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__41(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__42(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs42
    %obs43 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__42(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__43(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs43
    %obs44 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__43(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__44(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs44
    %obs45 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__44(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__45(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs45
    %obs46 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__45(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__46(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs46
    %obs47 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__46(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__47(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs47
    %obs48 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__47(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__48(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs48
    %obs49 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__48(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__49(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs49
    %obs50 = smt.forall {
    ^bb0(%obsarg0: !smt.int, %obsarg1: !smt.int):
      %obs51 = smt.apply_func %obsF__49(%obsarg0, %obsarg1) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obsc1 = smt.int.constant 1
      %obs52 = smt.int.add %obsarg0, %obsc1
      %obsc-65536 = smt.int.constant -65536
      %obs53 = smt.int.mod %obs52, %obsc-65536
      %obsc1_0 = smt.int.constant 1
      %obs54 = smt.int.add %obsarg1, %obsc1_0
      %obs55 = smt.apply_func %obsF__50(%obs53, %obs54) : !smt.func<(!smt.int, !smt.int) !smt.bool>
      %obstrue = smt.constant true
      %obs56 = smt.and %obs51, %obstrue
      %obs57 = smt.implies %obs56, %obs55
      smt.yield %obs57 : !smt.bool
    }
    smt.assert %obs50
  }
}

