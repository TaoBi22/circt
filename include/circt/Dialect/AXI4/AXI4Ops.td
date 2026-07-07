//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_TD
#define CIRCT_DIALECT_AXI4_AXI4OPS_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "circt/Dialect/AXI4/AXI4Types.td"
include "circt/Dialect/AXI4/AXI4Attributes.td"
include "circt/Dialect/AXI4/AXI4Interfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class AXI4Op<string mnemonic, list<Trait> traits = []> :
    Op<AXI4Dialect, mnemonic, traits>;

def NodeOp : AXI4Op<"node", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "An AXI4 node referencing a blackbox hardware module";
  let description = [{
    Describes an instantiated blackbox hardware module that has AXI4 ports.
    The symbol must resolve to an `hw.module`-like operation. The resulting
    `!axi4.node` value is used by `axi4.manager_port` and
    `axi4.subordinate_port` to refer to the instantiated module.

    Example:
    ```mlir
    %node = axi4.node @my_module : !axi4.node
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module);
  let results = (outs NodeType:$node);

  let assemblyFormat = "$module attr-dict `:` qualified(type($node))";
  let hasVerifier = 1;
}

def WindowArrayAttr : TypedArrayAttrBase<WindowAttr, "array of access windows">;

def ManagerPortOp : AXI4Op<"manager_port"> {
  let summary = "An AXI4 manager endpoint";
  let description = [{
    Describes a manager endpoint port. The `access` windows specify the
    regions this manager can send requests to, `outstanding_reads` and
    `outstanding_writes` bound the number of concurrent outstanding
    transactions, `clock` is the clock domain it operates in, and `node`
    optionally indicates the module this port is attached to (in which case
    `port_mapping` must also be given to indicate which port on that module
    this corresponds to). Returns the `!axi4.port` interface for the manager
    port.

    Example:
    ```mlir
    %manager = axi4.manager_port %clk node %node {
      port_mapping = #axi4.port_wires<"clk", "axi_in">,
      access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
      outstanding_reads = 4 : ui32,
      outstanding_writes = 4 : ui32
    } : !axi4.port<32, 64, 4>

    %nodeless_manager = axi4.manager_port %clk {
      access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
      outstanding_reads = 4 : ui32,
      outstanding_writes = 4 : ui32
    } : !axi4.port<32, 64, 4>
    ```
  }];

  let arguments = (ins ClockType:$clock,
                       Optional<NodeType>:$node,
                       OptionalAttr<AXI4PortMappingAttrInterface>:$port_mapping,
                       WindowArrayAttr:$access,
                       UI32Attr:$outstanding_reads,
                       UI32Attr:$outstanding_writes);
  let results = (outs PortType:$port);

  let assemblyFormat = "$clock (`node` $node^)? attr-dict `:` qualified(type($port))";
  let hasVerifier = 1;
}

def SubordinatePortOp : AXI4Op<"subordinate_port"> {
  let summary = "An AXI4 subordinate endpoint";
  let description = [{
    Describes a subordinate endpoint port. Takes the upstream `!axi4.port`,
    `clock` for the clock domain it operates in, an array of `access`
    windows it can handle requests to, the maximum number of
    `outstanding_requests` it can concurrently handle, and optionally `node`
    indicating the module this port is attached to (in which case
    `port_mapping` must also be given to indicate which port on that module
    this corresponds to).

    Example:
    ```mlir
    axi4.subordinate_port %manager, %clk node %node {
      port_mapping = #axi4.req_resp_structs<"clk", "axi_sub_req_i", "axi_sub_resp_o">,
      access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
      outstanding_requests = 4 : ui32
    } : !axi4.port<32, 64, 4>

    axi4.subordinate_port %nodeless_manager, %clk {
      access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
      outstanding_requests = 4 : ui32
    } : !axi4.port<32, 64, 4>
    ```
  }];

  let arguments = (ins PortType:$upstream,
                       ClockType:$clock,
                       Optional<NodeType>:$node,
                       OptionalAttr<AXI4PortMappingAttrInterface>:$port_mapping,
                       WindowArrayAttr:$access,
                       UI32Attr:$outstanding_requests);

  let assemblyFormat = "$upstream `,` $clock (`node` $node^)? attr-dict `:` qualified(type($upstream))";
  let hasVerifier = 1;
}

def XbarOp : AXI4Op<"xbar"> {
  let summary = "An AXI4 crossbar interconnect";
  let description = [{
    Describes a crossbar interconnect combining a variadic number of upstream
    `!axi4.port` connections and operating in the clock domain given by an
    `!axi4.clock`. All upstream ports must have the same type.

    Example:
    ```mlir
    %xbar = axi4.xbar %clk mgrs %manager1, %manager2 : (!axi4.port<32, 64, 4>, !axi4.port<32, 64, 4>) -> !axi4.port<32, 64, 5>
    ```
  }];

  let arguments = (ins ClockType:$clock, Variadic<PortType>:$upstream);
  let results = (outs PortType:$port);

  let assemblyFormat = "$clock `mgrs` $upstream attr-dict `:` functional-type($upstream, $port)";
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
