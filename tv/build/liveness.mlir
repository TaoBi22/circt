module {
  smt.solver() : () -> () {
    %obsc0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
    %obsc0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %obsc-1_bv16 = smt.bv.constant #smt.bv<-1> : !smt.bv<16>
    %obsc-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %obsc0_bv1_0 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %obsF_CTR_IDLE = smt.declare_fun "F_CTR_IDLE" : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
    %obsF_CTR_INCR = smt.declare_fun "F_CTR_INCR" : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
    %obsF_CTR_ERROR = smt.declare_fun "F_CTR_ERROR" : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
    %obs0 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<16>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<1>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>):
      %obsc0_bv16_1 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %obsc0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsc-1_bv1, %obsarg9, %obsc0_bv1_0, %obsc0_bv16_1, %obsc0_bv1_2, %obsc0_bv1_3) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      smt.yield %obs7 : !smt.bool
    }
    smt.assert %obs0
    %obs1 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>):
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs8 = smt.apply_func %obsF_CTR_INCR(%obsc0_bv1_0, %obsc0_bv1_0, %obsc-1_bv1, %obsc0_bv16, %obsc-1_bv1, %obsc0_bv1_0) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obsc-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs9 = smt.eq %obsarg0, %obsc-1_bv1_1 : !smt.bv<1>
      %obs10 = smt.and %obs7, %obs9
      %obs11 = smt.implies %obs10, %obs8
      smt.yield %obs11 : !smt.bool
    }
    smt.assert %obs1
    %obs2 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>):
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs8 = smt.apply_func %obsF_CTR_ERROR(%obsc-1_bv1, %obsc-1_bv1, %obsc0_bv1_0, %obsarg11, %obsarg12, %obsc-1_bv1) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs9 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs10 = smt.eq %obsarg0, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs11 = smt.ite %obs10, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obs12 = smt.bv.and %obs9, %obs11 : !smt.bv<1>
      %obsc-1_bv1_3 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs13 = smt.eq %obs9, %obsc-1_bv1_3 : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.eq %obsarg0, %obsc-1_bv1_4 : !smt.bv<1>
      %obs15 = smt.not %obs14
      %obs16 = smt.and %obs13, %obs15
      %obs17 = smt.and %obs7, %obs16
      %obs18 = smt.implies %obs17, %obs8
      smt.yield %obs18 : !smt.bool
    }
    smt.assert %obs2
    %obs3 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>):
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs8 = smt.apply_func %obsF_CTR_IDLE(%obsc-1_bv1, %obsc0_bv1_0, %obsc0_bv1_0, %obsarg11, %obsarg12, %obsc0_bv1_0) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs9 = smt.eq %obsarg0, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs10 = smt.ite %obs9, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obs11 = smt.eq %obsarg2, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs12 = smt.ite %obs11, %obsc-1_bv1_4, %obsc0_bv1_3 : !smt.bv<1>
      %obs13 = smt.eq %obsarg4, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_5 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_6 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.ite %obs13, %obsc-1_bv1_6, %obsc0_bv1_5 : !smt.bv<1>
      %obs15 = smt.bv.and %obs10, %obs12 : !smt.bv<1>
      %obs16 = smt.bv.and %obs15, %obs14 : !smt.bv<1>
      %obsc-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs17 = smt.eq %obs16, %obsc-1_bv1_7 : !smt.bv<1>
      %obsc-1_bv1_8 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs18 = smt.eq %obsarg0, %obsc-1_bv1_8 : !smt.bv<1>
      %obs19 = smt.not %obs18
      %obs20 = smt.and %obs17, %obs19
      %obs21 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs22 = smt.eq %obsarg0, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_9 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_10 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs23 = smt.ite %obs22, %obsc-1_bv1_10, %obsc0_bv1_9 : !smt.bv<1>
      %obs24 = smt.bv.and %obs21, %obs23 : !smt.bv<1>
      %obsc-1_bv1_11 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs25 = smt.eq %obs21, %obsc-1_bv1_11 : !smt.bv<1>
      %obs26 = smt.not %obs25
      %obs27 = smt.and %obs20, %obs26
      %obs28 = smt.and %obs7, %obs27
      %obs29 = smt.implies %obs28, %obs8
      smt.yield %obs29 : !smt.bool
    }
    smt.assert %obs3
    %obs4 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>):
      %obs7 = smt.apply_func %obsF_CTR_INCR(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs8 = smt.bv.cmp slt %obsarg6, %obsc0_bv16 : !smt.bv<16>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs9 = smt.ite %obs8, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs10 = smt.bv.add %obsarg11, %obsc1_bv16 : !smt.bv<16>
      %obs11 = smt.apply_func %obsF_CTR_IDLE(%obsc-1_bv1, %obsc0_bv1_0, %obsc0_bv1_0, %obs10, %obs9, %obsc0_bv1_0) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs12 = smt.eq %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs13 = smt.ite %obs12, %obsc-1_bv1_4, %obsc0_bv1_3 : !smt.bv<1>
      %obsc-1_bv1_5 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.eq %obs13, %obsc-1_bv1_5 : !smt.bv<1>
      %obs15 = smt.and %obs7, %obs14
      %obs16 = smt.implies %obs15, %obs11
      smt.yield %obs16 : !smt.bool
    }
    smt.assert %obs4
    %obs5 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>):
      %obs7 = smt.apply_func %obsF_CTR_INCR(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs8 = smt.bv.cmp slt %obsarg6, %obsc0_bv16 : !smt.bv<16>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs9 = smt.ite %obs8, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs10 = smt.bv.add %obsarg11, %obsc1_bv16 : !smt.bv<16>
      %obs11 = smt.apply_func %obsF_CTR_INCR(%obsc0_bv1_0, %obsc0_bv1_0, %obsc-1_bv1, %obs10, %obs9, %obsc0_bv1_0) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs12 = smt.distinct %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs13 = smt.ite %obs12, %obsc-1_bv1_4, %obsc0_bv1_3 : !smt.bv<1>
      %obs14 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs15 = smt.bv.xor %obs14, %obsc-1_bv1 : !smt.bv<1>
      %obs16 = smt.bv.and %obs13, %obs15 : !smt.bv<1>
      %obsc-1_bv1_5 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs17 = smt.eq %obs16, %obsc-1_bv1_5 : !smt.bv<1>
      %obs18 = smt.eq %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_6 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs19 = smt.ite %obs18, %obsc-1_bv1_7, %obsc0_bv1_6 : !smt.bv<1>
      %obsc-1_bv1_8 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs20 = smt.eq %obs19, %obsc-1_bv1_8 : !smt.bv<1>
      %obs21 = smt.not %obs20
      %obs22 = smt.and %obs17, %obs21
      %obs23 = smt.and %obs7, %obs22
      %obs24 = smt.implies %obs23, %obs11
      smt.yield %obs24 : !smt.bool
    }
    smt.assert %obs5
    %obs6 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>):
      %obs7 = smt.apply_func %obsF_CTR_INCR(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs8 = smt.apply_func %obsF_CTR_ERROR(%obsc-1_bv1, %obsc-1_bv1, %obsc0_bv1_0, %obsarg11, %obsarg12, %obsc-1_bv1) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>) !smt.bool>
      %obs9 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obsc-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs10 = smt.eq %obs9, %obsc-1_bv1_1 : !smt.bv<1>
      %obs11 = smt.eq %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_3 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs12 = smt.ite %obs11, %obsc-1_bv1_3, %obsc0_bv1_2 : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs13 = smt.eq %obs12, %obsc-1_bv1_4 : !smt.bv<1>
      %obs14 = smt.not %obs13
      %obs15 = smt.and %obs10, %obs14
      %obs16 = smt.distinct %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_5 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_6 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs17 = smt.ite %obs16, %obsc-1_bv1_6, %obsc0_bv1_5 : !smt.bv<1>
      %obs18 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs19 = smt.bv.xor %obs18, %obsc-1_bv1 : !smt.bv<1>
      %obs20 = smt.bv.and %obs17, %obs19 : !smt.bv<1>
      %obsc-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs21 = smt.eq %obs20, %obsc-1_bv1_7 : !smt.bv<1>
      %obs22 = smt.not %obs21
      %obs23 = smt.and %obs15, %obs22
      %obs24 = smt.and %obs7, %obs23
      %obs25 = smt.implies %obs24, %obs8
      smt.yield %obs25 : !smt.bool
    }
    smt.assert %obs6
  }
}

