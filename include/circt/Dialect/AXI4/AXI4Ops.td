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
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class AXI4Op<string mnemonic, list<Trait> traits = []> :
    Op<AXI4Dialect, mnemonic, traits>;

def WindowArrayAttr : TypedArrayAttrBase<WindowAttr, "array of access windows">;

def ManagerOp : AXI4Op<"manager", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "An AXI4 manager endpoint";
  let description = [{
    Describes a manager endpoint. The `access` windows specify the regions this
    manager can send requests to, `outstanding_reads` and `outstanding_writes`
    bound the number of concurrent outstanding transactions, and the symbol
    indicates the `hw.module` this manager instantiates. Returns the
    `!axi4.port` interface for the manager.

    Example:
    ```mlir
    %manager = axi4.manager @manager_core {
      access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
      outstanding_reads = 4 : ui32,
      outstanding_writes = 4 : ui32
    } : !axi4.port<32, 64, 4>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$module,
                       WindowArrayAttr:$access,
                       UI32Attr:$outstanding_reads,
                       UI32Attr:$outstanding_writes);
  let results = (outs PortType:$port);

  let assemblyFormat = "$module attr-dict `:` qualified(type($port))";
  let hasVerifier = 1;
}

def SubordinateOp : AXI4Op<"subordinate", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "An AXI4 subordinate endpoint";
  let description = [{
    Describes a subordinate endpoint. Takes the upstream `!axi4.port`, an array
    of `access` windows it can handle requests to, the maximum number of
    `outstanding_requests` it can concurrently handle, and a symbol indicating
    the `hw.module` this subordinate instantiates.

    Example:
    ```mlir
    axi4.subordinate %manager @subordinate_peripheral {
      access = [#axi4.window<base = 0, size = 4096, burst_specs = [<fixed>]>],
      outstanding_requests = 4 : ui32
    } : !axi4.port<32, 64, 4>
    ```
  }];

  let arguments = (ins PortType:$upstream,
                       FlatSymbolRefAttr:$module,
                       WindowArrayAttr:$access,
                       UI32Attr:$outstanding_requests);

  let assemblyFormat = "$upstream $module attr-dict `:` qualified(type($upstream))";
  let hasVerifier = 1;
}

def XbarOp : AXI4Op<"xbar", [SameOperandsAndResultType]> {
  let summary = "An AXI4 crossbar interconnect";
  let description = [{
    Describes a crossbar interconnect combining a variadic number of upstream
    `!axi4.port` connections. All operands must have the same type as the
    result.

    Example:
    ```mlir
    %xbar = axi4.xbar %manager1, %manager2 : !axi4.port<32, 64, 4>
    ```
  }];

  let arguments = (ins Variadic<PortType>:$upstream);
  let results = (outs PortType:$port);

  let assemblyFormat = "$upstream attr-dict `:` qualified(type($port))";
}

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
