// RUN: circt-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-error @below {{'fixed' burst kind cannot have a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<fixed, len = 4>} : () -> ()

// -----

// expected-error @below {{'incr' burst kind requires a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<incr>} : () -> ()

// -----

// expected-error @below {{'wrap' burst kind requires a 'len'}}
"test.attrs"() {a = #axi4.burst_spec<wrap>} : () -> ()

// -----

// expected-error @below {{references unknown symbol @does_not_exist}}
%node = axi4.node @does_not_exist : !axi4.node

// -----

func.func private @not_a_module()
// expected-error @below {{symbol @not_a_module must refer to an 'hw.module'}}
%node = axi4.node @not_a_module : !axi4.node

// -----

hw.module.extern @mgr_module()
%node = axi4.node @mgr_module : !axi4.node
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{'node' and 'port_mapping' must either both be given or both be omitted}}
%mgr = axi4.manager_port %clk, %rst node %node {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{'node' and 'port_mapping' must either both be given or both be omitted}}
%mgr = axi4.manager_port %clk, %rst {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{access windows overlap}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>,
            #axi4.window<base = 2048, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// 'outstanding_reads' of 32 exceeds 2^4 = 16 addressable by the read ID width.
// expected-error @below {{outstanding_reads (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 32 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// 'outstanding_writes' of 32 exceeds 2^4 = 16 addressable by the write ID width.
// expected-error @below {{outstanding_writes (32) exceeds the maximum of 2^4 (16)}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 32 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{xbar return type's write id width must be at least the input write id width + ceil(log2(number of managers)) (i.e., 5)}}
%xbar = "axi4.xbar"(%clk, %rst, %a, %b) : (!axi4.clock, !axi4.reset, !axi4.port<32, 64, 4, 4, 0>, !axi4.port<32, 64, 4, 4, 0>) -> !axi4.port<32, 64, 4, 4, 0>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{xbar return type's read id width must be at least the input read id width + ceil(log2(number of managers)) (i.e., 5)}}
%xbar = "axi4.xbar"(%clk, %rst, %a, %b) : (!axi4.clock, !axi4.reset, !axi4.port<32, 64, 4, 4, 0>, !axi4.port<32, 64, 4, 4, 0>) -> !axi4.port<32, 64, 5, 4, 0>

// -----

%a = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%b = unrealized_conversion_cast to !axi4.port<32, 64, 5, 5, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{all upstream ports must have the same type}}
%xbar = "axi4.xbar"(%clk, %rst, %a, %b) : (!axi4.clock, !axi4.reset, !axi4.port<32, 64, 4, 4, 0>, !axi4.port<32, 64, 5, 5, 0>) -> !axi4.port<32, 64, 5, 5, 0>

// -----

%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{result must have at most one use}}
%mgr = axi4.manager_port %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

hw.module.extern @combo_module()
// expected-error @below {{ports sharing the clock port 'clk' must share the same '!axi4.clock' operand}}
%node = axi4.node @combo_module : !axi4.node
%clkA = unrealized_conversion_cast to !axi4.clock
%clkB = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
%mgr = axi4.manager_port %clkA, %rst node %node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clkB, %rst node %node {
  port_mapping = #axi4.req_resp_structs<"clk", "rst_ni", "axi_sub_req_i", "axi_sub_resp_o">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

hw.module.extern @combo_module()
// expected-error @below {{ports sharing the reset port 'rst_ni' must share the same '!axi4.reset' operand}}
%node = axi4.node @combo_module : !axi4.node
%clk = unrealized_conversion_cast to !axi4.clock
%rstA = unrealized_conversion_cast to !axi4.reset
%rstB = unrealized_conversion_cast to !axi4.reset
%mgr = axi4.manager_port %clk, %rstA node %node {
  port_mapping = #axi4.port_wires<"clk", "rst_ni", "axi_in">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_reads = 4 : ui32,
  outstanding_writes = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %mgr, %clk, %rstB node %node {
  port_mapping = #axi4.req_resp_structs<"clk", "rst_ni", "axi_sub_req_i", "axi_sub_resp_o">,
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%port = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{'axi4.cut' op result must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
%cut = axi4.cut %clk, %rst at %port : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %cut, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %cut, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%port = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0>
%clkA = unrealized_conversion_cast to !axi4.clock
%rstA = unrealized_conversion_cast to !axi4.reset
%clkB = unrealized_conversion_cast to !axi4.clock
%rstB = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{'axi4.cdc' op result must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
%cdc = axi4.cdc %port from [%clkA, %rstA] to [%clkB, %rstB] : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %cdc, %clkB, %rstB {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %cdc, %clkB, %rstB {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>

// -----

%port = unrealized_conversion_cast to !axi4.port<32, 32, 4, 4, 0>
%clk = unrealized_conversion_cast to !axi4.clock
%rst = unrealized_conversion_cast to !axi4.reset
// expected-error @below {{'axi4.data_width_converter' op result must have at most one use; route through an 'axi4.xbar' to fan out to multiple endpoints}}
%dwc = axi4.data_width_converter %clk, %rst, %port : (!axi4.port<32, 32, 4, 4, 0>) -> !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %dwc, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
axi4.subordinate_port %dwc, %clk, %rst {
  access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
  outstanding_requests = 4 : ui32
} : !axi4.port<32, 64, 4, 4, 0>
