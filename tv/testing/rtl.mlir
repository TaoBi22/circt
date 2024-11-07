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
    %state_reg = seq.compreg sym @state_reg %41, %clk : i4  
    %c0_i16 = hw.constant 0 : i16
    %x0 = seq.compreg sym @x0 %38, %clk : i16  
    %c1_i16 = hw.constant 1 : i16
    %0 = comb.icmp eq %state_reg, %c0_i4 : i4
    %1 = comb.add %x0, %c1_i16 : i16
    %2 = comb.mux %0, %1, %x0 : i16
    %3 = comb.mux %0, %c1_i4, %state_reg : i4
    %4 = comb.icmp eq %state_reg, %c1_i4 : i4
    %5 = comb.add %x0, %c1_i16 : i16
    %6 = comb.mux %4, %5, %2 : i16
    %7 = comb.mux %4, %c2_i4, %3 : i4
    %8 = comb.icmp eq %state_reg, %c2_i4 : i4
    %9 = comb.add %x0, %c1_i16 : i16
    %10 = comb.mux %8, %9, %6 : i16
    %11 = comb.mux %8, %c3_i4, %7 : i4
    %12 = comb.icmp eq %state_reg, %c3_i4 : i4
    %13 = comb.add %x0, %c1_i16 : i16
    %14 = comb.mux %12, %13, %10 : i16
    %15 = comb.mux %12, %c4_i4, %11 : i4
    %16 = comb.icmp eq %state_reg, %c4_i4 : i4
    %17 = comb.add %x0, %c1_i16 : i16
    %18 = comb.mux %16, %17, %14 : i16
    %19 = comb.mux %16, %c5_i4, %15 : i4
    %20 = comb.icmp eq %state_reg, %c5_i4 : i4
    %21 = comb.add %x0, %c1_i16 : i16
    %22 = comb.mux %20, %21, %18 : i16
    %23 = comb.mux %20, %c6_i4, %19 : i4
    %24 = comb.icmp eq %state_reg, %c6_i4 : i4
    %25 = comb.add %x0, %c1_i16 : i16
    %26 = comb.mux %24, %25, %22 : i16
    %27 = comb.mux %24, %c7_i4, %23 : i4
    %28 = comb.icmp eq %state_reg, %c7_i4 : i4
    %29 = comb.add %x0, %c1_i16 : i16
    %30 = comb.mux %28, %29, %26 : i16
    %31 = comb.mux %28, %c-8_i4, %27 : i4
    %32 = comb.icmp eq %state_reg, %c-8_i4 : i4
    %33 = comb.add %x0, %c1_i16 : i16
    %34 = comb.mux %32, %33, %30 : i16
    %35 = comb.mux %32, %c-7_i4, %31 : i4
    %36 = comb.icmp eq %state_reg, %c-7_i4 : i4
    %37 = comb.add %x0, %c1_i16 : i16
    %38 = comb.mux %36, %37, %34 : i16
    %39 = comb.mux %36, %c-6_i4, %35 : i4
    %40 = comb.icmp eq %state_reg, %c-6_i4 : i4
    %41 = comb.mux %40, %c-6_i4, %39 : i4
    %c1_i32 = hw.constant 1 : i32
    %time_reg = seq.compreg sym @time_reg %added, %clk : i32
    %added = comb.add %time_reg, %c1_i32 : i32
    hw.output
  }
}

