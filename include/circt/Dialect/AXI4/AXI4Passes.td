//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4PASSES_TD
#define CIRCT_DIALECT_AXI4_AXI4PASSES_TD

include "mlir/Pass/PassBase.td"

def VerifyAXI4Networks : Pass<"verify-axi4-networks", "mlir::ModuleOp"> {
  let summary = "Globally verify AXI4 networks";
  let description = [{
    Performs global verifications of AXI4 networks that are too expensive to
    include in individual operation verifiers, e.g. that reachable addresses
    cannot be reached multiple ways, that manager access windows fully map to
    downstream endpoints with compatible burst kinds, and that the network is
    loop-free.
  }];
  let dependentDialects = ["circt::axi4::AXI4Dialect"];
}

#endif // CIRCT_DIALECT_AXI4_AXI4PASSES_TD
