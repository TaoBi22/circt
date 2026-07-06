//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the types for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4TYPES_TD
#define CIRCT_DIALECT_AXI4_AXI4TYPES_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class AXI4TypeDef<string name, list<Trait> traits = []> :
    TypeDef<AXI4Dialect, name, traits>;

def ClockType : AXI4TypeDef<"Clock"> {
  let mnemonic = "clock";
  let summary = "A clock domain over AXI4 components";
  let description = [{
    Defines a clock domain over AXI4 components.
  }];
}

def PortType : AXI4TypeDef<"Port"> {
  let mnemonic = "port";
  let summary = "An AXI4 port interface";
  let description = [{
    The primary interface passed between managers, subordinates, and routing &
    connection primitives. The address, data, and ID widths are carried in the
    type, e.g. `!axi4.port<32, 64, 4>`.
  }];

  let parameters = (ins "uint32_t":$addressWidth,
                        "uint32_t":$dataWidth,
                        "uint32_t":$idWidth);

  let assemblyFormat = "`<` $addressWidth `,` $dataWidth `,` $idWidth `>`";
}

#endif // CIRCT_DIALECT_AXI4_AXI4TYPES_TD
