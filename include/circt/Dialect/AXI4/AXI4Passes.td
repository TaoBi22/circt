//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the passes for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4PASSES_TD
#define CIRCT_DIALECT_AXI4_AXI4PASSES_TD

include "mlir/Pass/PassBase.td"

def VerifyAXI4Networks : Pass<"verify-axi4-networks", "mlir::ModuleOp"> {
  let summary = "Globally verify AXI4 networks";
  let description = [{
    Verifies non-local properties of an AXI4 network:
    - Ensures every `!axi4.port` value has at most one consumer, and that ports
    do not connect endpoints in different clock or reset domains.
    - Ensures a network occupies a single clock and reset domain.
    - Warns where an `axi4.xbar` downstream port can hold fewer outstanding
    requests than the managers reaching it can issue. This costs throughput
    rather than correctness, since AXI4 flow control stalls the manager rather
    than overrunning the subordinate.
  }];

  let dependentDialects = ["circt::axi4::AXI4Dialect"];
}

def PruneAXI4Networks : Pass<"prune-axi4-networks", "mlir::ModuleOp"> {
  let summary = "Prune the unreachable parts of AXI4 networks";
  let description = [{
    Removes the downstream ports of `axi4.xbar` and `axi4.demux` operations
    that no upstream manager can address, along with the ops downstream of them.

    Where such a port feeds something this pass will not erase - an interface
    bridged out into HW logic, a memory interface driving live logic, or another
    routing operation - it warns and leaves the branch alone.
  }];

  let dependentDialects = ["circt::axi4::AXI4Dialect"];
}

def LowerAXI4DummiesToAXI : Pass<"lower-axi4-dummies-to-axi", "mlir::ModuleOp"> {
  let summary = "Lower the dummies subdialect to the AXI4 dialect";
  let description = [{
    Rewrites a network described with the `axi4.dummies` operations into the
    AXI4 dialect, inferring the parameterisation they leave out: the access
    windows of each manager from the accesses it declares, the ID widths from
    the outstanding request counts, and the placement of the width converters
    needed where an endpoint and the interconnect it connects to disagree.

    The external endpoints become ports of the module the network is described
    in, so that module must not be instantiated.
  }];

  let dependentDialects = ["circt::axi4::AXI4Dialect", "circt::hw::HWDialect"];
}

#endif // CIRCT_DIALECT_AXI4_AXI4PASSES_TD
