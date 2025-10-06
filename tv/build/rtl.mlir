module {
  hw.module @fsm10(in %clk : !seq.clock, in %rst : i1) {
    %c0_i4 = hw.constant 0 : i4
    %c1_i4 = hw.constant 1 : i4
    %c2_i4 = hw.constant 2 : i4
    %c3_i4 = hw.constant 3 : i4
    %c4_i4 = hw.constant 4 : i4
    %c5_i4 = hw.constant 5 : i4
    %c6_i4 = hw.constant 6 : i4
    %c7_i4 = hw.constant 7 : i4
    %c-8_i4 = hw.constant -8 : i4
    %c-7_i4 = hw.constant -7 : i4
    %c-6_i4 = hw.constant -6 : i4
    %0 = seq.initial() {
      %c0_i4_0 = hw.constant 0 : i4
      seq.yield %c0_i4_0 : i4
    } : () -> !seq.immutable<i4>
    %state_reg = seq.compreg sym @state_reg %43, %clk initial %0 : i4  
    %c0_i16 = hw.constant 0 : i16
    %1 = seq.initial() {
      %c0_i16_0 = hw.constant 0 : i16
      seq.yield %c0_i16_0 : i16
    } : () -> !seq.immutable<i16>
    %x0 = seq.compreg sym @x0 %40, %clk initial %1 : i16  
    %c1_i16 = hw.constant 1 : i16
    %2 = comb.icmp eq %state_reg, %c0_i4 : i4
    %3 = comb.add %x0, %c1_i16 : i16
    %4 = comb.mux %2, %3, %x0 : i16
    %5 = comb.mux %2, %c1_i4, %state_reg : i4
    %6 = comb.icmp eq %state_reg, %c1_i4 : i4
    %7 = comb.add %x0, %c1_i16 : i16
    %8 = comb.mux %6, %7, %4 : i16
    %9 = comb.mux %6, %c2_i4, %5 : i4
    %10 = comb.icmp eq %state_reg, %c2_i4 : i4
    %11 = comb.add %x0, %c1_i16 : i16
    %12 = comb.mux %10, %11, %8 : i16
    %13 = comb.mux %10, %c3_i4, %9 : i4
    %14 = comb.icmp eq %state_reg, %c3_i4 : i4
    %15 = comb.add %x0, %c1_i16 : i16
    %16 = comb.mux %14, %15, %12 : i16
    %17 = comb.mux %14, %c4_i4, %13 : i4
    %18 = comb.icmp eq %state_reg, %c4_i4 : i4
    %19 = comb.add %x0, %c1_i16 : i16
    %20 = comb.mux %18, %19, %16 : i16
    %21 = comb.mux %18, %c5_i4, %17 : i4
    %22 = comb.icmp eq %state_reg, %c5_i4 : i4
    %23 = comb.add %x0, %c1_i16 : i16
    %24 = comb.mux %22, %23, %20 : i16
    %25 = comb.mux %22, %c6_i4, %21 : i4
    %26 = comb.icmp eq %state_reg, %c6_i4 : i4
    %27 = comb.add %x0, %c1_i16 : i16
    %28 = comb.mux %26, %27, %24 : i16
    %29 = comb.mux %26, %c7_i4, %25 : i4
    %30 = comb.icmp eq %state_reg, %c7_i4 : i4
    %31 = comb.add %x0, %c1_i16 : i16
    %32 = comb.mux %30, %31, %28 : i16
    %33 = comb.mux %30, %c-8_i4, %29 : i4
    %34 = comb.icmp eq %state_reg, %c-8_i4 : i4
    %35 = comb.add %x0, %c1_i16 : i16
    %36 = comb.mux %34, %35, %32 : i16
    %37 = comb.mux %34, %c-7_i4, %33 : i4
    %38 = comb.icmp eq %state_reg, %c-7_i4 : i4
    %39 = comb.add %x0, %c1_i16 : i16
    %40 = comb.mux %38, %39, %36 : i16
    %41 = comb.mux %38, %c-6_i4, %37 : i4
    %42 = comb.icmp eq %state_reg, %c-6_i4 : i4
    %43 = comb.mux %42, %c-6_i4, %41 : i4
    %timer_init = seq.initial() {
%c0_i16_0_in = hw.constant 0 : i32
seq.yield %c0_i16_0_in : i32
} : () -> !seq.immutable<i32>
%mySpecialConstant = hw.constant 2 : i32
%time_reg = seq.compreg sym @time_reg %added, %clk initial %timer_init : i32
%added = comb.add %time_reg, %mySpecialConstant : i32
    hw.output
}}