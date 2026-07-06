//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes attribute interfaces for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4INTERFACES_TD
#define CIRCT_DIALECT_AXI4_AXI4INTERFACES_TD

include "mlir/IR/Interfaces.td"
include "mlir/IR/OpBase.td"

def AXI4PortMappingAttrInterface : AttrInterface<"AXI4PortMappingAttrInterface"> {
  let description = [{
    Common interface for attributes that describe how an `axi4.manager_port`
    or `axi4.subordinate_port` maps onto the ports of the RTL module it is
    attached to via `axi4.node`. Implementations describe a style of RTL
    lowering (e.g. a single struct port, split req/resp struct ports, or an
    interface port), and must report the names of the module ports that the
    `!axi4.clock` and `!axi4.reset` operands of the owning port should be
    routed to.
  }];
  let cppNamespace = "::circt::axi4";

  let methods = [
    InterfaceMethod<[{
      Returns the name of the module port that the `!axi4.clock` operand of
      the owning `manager_port`/`subordinate_port` should be routed to.
    }], "::llvm::StringRef", "getClockPort">,
    InterfaceMethod<[{
      Returns the name of the module port that the `!axi4.reset` operand of
      the owning `manager_port`/`subordinate_port` should be routed to.
    }], "::llvm::StringRef", "getResetPort">,
  ];
}

#endif // CIRCT_DIALECT_AXI4_AXI4INTERFACES_TD
