module {
  smt.solver() : () -> () {
    %obsc0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
    %obsc0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %obsc-1_bv16 = smt.bv.constant #smt.bv<-1> : !smt.bv<16>
    %obsc-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
    %obsc0_bv1_0 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
    %obsF_CTR_IDLE = smt.declare_fun "F_CTR_IDLE" : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
    %obsF_CTR_INCR = smt.declare_fun "F_CTR_INCR" : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
    %obsF_CTR_ERROR = smt.declare_fun "F_CTR_ERROR" : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
    %obs0 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<16>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<1>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<8>):
      %obsc0_bv16_1 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
      %obsc0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc0_bv8 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsc-1_bv1, %obsarg9, %obsc0_bv1_0, %obsc0_bv16_1, %obsc0_bv1_2, %obsc0_bv1_3, %obsc0_bv8) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      smt.yield %obs7 : !smt.bool
    }
    smt.assert %obs0
    %obs1 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>, %obsarg14: !smt.bv<8>):
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13, %obsarg14) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obsc1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
      %obs8 = smt.bv.add %obsarg14, %obsc1_bv8 : !smt.bv<8>
      %obs9 = smt.apply_func %obsF_CTR_INCR(%obsc0_bv1_0, %obsc0_bv1_0, %obsc-1_bv1, %obsc0_bv16, %obsc-1_bv1, %obsc0_bv1_0, %obs8) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obsc-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs10 = smt.eq %obsarg0, %obsc-1_bv1_1 : !smt.bv<1>
      %obs11 = smt.and %obs7, %obs10
      %obs12 = smt.implies %obs11, %obs9
      smt.yield %obs12 : !smt.bool
    }
    smt.assert %obs1
    %obs2 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>, %obsarg14: !smt.bv<8>):
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13, %obsarg14) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obsc1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
      %obs8 = smt.bv.add %obsarg14, %obsc1_bv8 : !smt.bv<8>
      %obs9 = smt.apply_func %obsF_CTR_ERROR(%obsc-1_bv1, %obsc-1_bv1, %obsc0_bv1_0, %obsarg11, %obsarg12, %obsc-1_bv1, %obs8) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs10 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs11 = smt.eq %obsarg0, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs12 = smt.ite %obs11, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obs13 = smt.bv.and %obs10, %obs12 : !smt.bv<1>
      %obsc-1_bv1_3 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.eq %obs10, %obsc-1_bv1_3 : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs15 = smt.eq %obsarg0, %obsc-1_bv1_4 : !smt.bv<1>
      %obs16 = smt.not %obs15
      %obs17 = smt.and %obs14, %obs16
      %obs18 = smt.and %obs7, %obs17
      %obs19 = smt.implies %obs18, %obs9
      smt.yield %obs19 : !smt.bool
    }
    smt.assert %obs2
    %obs3 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>, %obsarg14: !smt.bv<8>):
      %obs7 = smt.apply_func %obsF_CTR_IDLE(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13, %obsarg14) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obsc1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
      %obs8 = smt.bv.add %obsarg14, %obsc1_bv8 : !smt.bv<8>
      %obs9 = smt.apply_func %obsF_CTR_IDLE(%obsc-1_bv1, %obsc0_bv1_0, %obsc0_bv1_0, %obsarg11, %obsarg12, %obsc0_bv1_0, %obs8) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs10 = smt.eq %obsarg0, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs11 = smt.ite %obs10, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obs12 = smt.eq %obsarg2, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs13 = smt.ite %obs12, %obsc-1_bv1_4, %obsc0_bv1_3 : !smt.bv<1>
      %obs14 = smt.eq %obsarg4, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_5 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_6 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs15 = smt.ite %obs14, %obsc-1_bv1_6, %obsc0_bv1_5 : !smt.bv<1>
      %obs16 = smt.bv.and %obs11, %obs13 : !smt.bv<1>
      %obs17 = smt.bv.and %obs16, %obs15 : !smt.bv<1>
      %obsc-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs18 = smt.eq %obs17, %obsc-1_bv1_7 : !smt.bv<1>
      %obsc-1_bv1_8 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs19 = smt.eq %obsarg0, %obsc-1_bv1_8 : !smt.bv<1>
      %obs20 = smt.not %obs19
      %obs21 = smt.and %obs18, %obs20
      %obs22 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs23 = smt.eq %obsarg0, %obsc0_bv1 : !smt.bv<1>
      %obsc0_bv1_9 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_10 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs24 = smt.ite %obs23, %obsc-1_bv1_10, %obsc0_bv1_9 : !smt.bv<1>
      %obs25 = smt.bv.and %obs22, %obs24 : !smt.bv<1>
      %obsc-1_bv1_11 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs26 = smt.eq %obs22, %obsc-1_bv1_11 : !smt.bv<1>
      %obs27 = smt.not %obs26
      %obs28 = smt.and %obs21, %obs27
      %obs29 = smt.and %obs7, %obs28
      %obs30 = smt.implies %obs29, %obs9
      smt.yield %obs30 : !smt.bool
    }
    smt.assert %obs3
    %obs4 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>, %obsarg14: !smt.bv<8>):
      %obs7 = smt.apply_func %obsF_CTR_INCR(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13, %obsarg14) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs8 = smt.bv.cmp slt %obsarg6, %obsc0_bv16 : !smt.bv<16>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs9 = smt.ite %obs8, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs10 = smt.bv.add %obsarg11, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
      %obs11 = smt.bv.add %obsarg14, %obsc1_bv8 : !smt.bv<8>
      %obs12 = smt.apply_func %obsF_CTR_IDLE(%obsc-1_bv1, %obsc0_bv1_0, %obsc0_bv1_0, %obs10, %obs9, %obsc0_bv1_0, %obs11) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs13 = smt.eq %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.ite %obs13, %obsc-1_bv1_4, %obsc0_bv1_3 : !smt.bv<1>
      %obsc-1_bv1_5 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs15 = smt.eq %obs14, %obsc-1_bv1_5 : !smt.bv<1>
      %obs16 = smt.and %obs7, %obs15
      %obs17 = smt.implies %obs16, %obs12
      smt.yield %obs17 : !smt.bool
    }
    smt.assert %obs4
    %obs5 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>, %obsarg14: !smt.bv<8>):
      %obs7 = smt.apply_func %obsF_CTR_INCR(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13, %obsarg14) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs8 = smt.bv.cmp slt %obsarg6, %obsc0_bv16 : !smt.bv<16>
      %obsc0_bv1_1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_2 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs9 = smt.ite %obs8, %obsc-1_bv1_2, %obsc0_bv1_1 : !smt.bv<1>
      %obsc1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
      %obs10 = smt.bv.add %obsarg11, %obsc1_bv16 : !smt.bv<16>
      %obsc1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
      %obs11 = smt.bv.add %obsarg14, %obsc1_bv8 : !smt.bv<8>
      %obs12 = smt.apply_func %obsF_CTR_INCR(%obsc0_bv1_0, %obsc0_bv1_0, %obsc-1_bv1, %obs10, %obs9, %obsc0_bv1_0, %obs11) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs13 = smt.distinct %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_3 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.ite %obs13, %obsc-1_bv1_4, %obsc0_bv1_3 : !smt.bv<1>
      %obs15 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs16 = smt.bv.xor %obs15, %obsc-1_bv1 : !smt.bv<1>
      %obs17 = smt.bv.and %obs14, %obs16 : !smt.bv<1>
      %obsc-1_bv1_5 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs18 = smt.eq %obs17, %obsc-1_bv1_5 : !smt.bv<1>
      %obs19 = smt.eq %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_6 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs20 = smt.ite %obs19, %obsc-1_bv1_7, %obsc0_bv1_6 : !smt.bv<1>
      %obsc-1_bv1_8 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs21 = smt.eq %obs20, %obsc-1_bv1_8 : !smt.bv<1>
      %obs22 = smt.not %obs21
      %obs23 = smt.and %obs18, %obs22
      %obs24 = smt.and %obs7, %obs23
      %obs25 = smt.implies %obs24, %obs12
      smt.yield %obs25 : !smt.bool
    }
    smt.assert %obs5
    %obs6 = smt.forall {
    ^bb0(%obsarg0: !smt.bv<1>, %obsarg1: !smt.bv<1>, %obsarg2: !smt.bv<1>, %obsarg3: !smt.bv<1>, %obsarg4: !smt.bv<1>, %obsarg5: !smt.bv<1>, %obsarg6: !smt.bv<16>, %obsarg7: !smt.bv<16>, %obsarg8: !smt.bv<1>, %obsarg9: !smt.bv<1>, %obsarg10: !smt.bv<1>, %obsarg11: !smt.bv<16>, %obsarg12: !smt.bv<1>, %obsarg13: !smt.bv<1>, %obsarg14: !smt.bv<8>):
      %obs7 = smt.apply_func %obsF_CTR_INCR(%obsarg8, %obsarg9, %obsarg10, %obsarg11, %obsarg12, %obsarg13, %obsarg14) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obsc1_bv8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
      %obs8 = smt.bv.add %obsarg14, %obsc1_bv8 : !smt.bv<8>
      %obs9 = smt.apply_func %obsF_CTR_ERROR(%obsc-1_bv1, %obsc-1_bv1, %obsc0_bv1_0, %obsarg11, %obsarg12, %obsc-1_bv1, %obs8) : !smt.func<(!smt.bv<1>, !smt.bv<1>, !smt.bv<1>, !smt.bv<16>, !smt.bv<1>, !smt.bv<1>, !smt.bv<8>) !smt.bool>
      %obs10 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obsc-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs11 = smt.eq %obs10, %obsc-1_bv1_1 : !smt.bv<1>
      %obs12 = smt.eq %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_2 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_3 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs13 = smt.ite %obs12, %obsc-1_bv1_3, %obsc0_bv1_2 : !smt.bv<1>
      %obsc-1_bv1_4 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs14 = smt.eq %obs13, %obsc-1_bv1_4 : !smt.bv<1>
      %obs15 = smt.not %obs14
      %obs16 = smt.and %obs11, %obs15
      %obs17 = smt.distinct %obsc-1_bv16, %obsarg11 : !smt.bv<16>
      %obsc0_bv1_5 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
      %obsc-1_bv1_6 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs18 = smt.ite %obs17, %obsc-1_bv1_6, %obsc0_bv1_5 : !smt.bv<1>
      %obs19 = smt.bv.or %obsarg2, %obsarg4 : !smt.bv<1>
      %obs20 = smt.bv.xor %obs19, %obsc-1_bv1 : !smt.bv<1>
      %obs21 = smt.bv.and %obs18, %obs20 : !smt.bv<1>
      %obsc-1_bv1_7 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
      %obs22 = smt.eq %obs21, %obsc-1_bv1_7 : !smt.bv<1>
      %obs23 = smt.not %obs22
      %obs24 = smt.and %obs16, %obs23
      %obs25 = smt.and %obs7, %obs24
      %obs26 = smt.implies %obs25, %obs9
      smt.yield %obs26 : !smt.bool
    }
    smt.assert %obs6
  }
}

