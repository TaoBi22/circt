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
}

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
