//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace axi4 {
#define GEN_PASS_DEF_VERIFYAXI4NETWORKS
#include "circt/Dialect/AXI4/AXI4Passes.h.inc"
} // namespace axi4
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;

namespace {
struct VerifyAXI4NetworksPass
    : public circt::axi4::impl::VerifyAXI4NetworksBase<VerifyAXI4NetworksPass> {
  void runOnOperation() override {
    // TODO: Implement global AXI4 network verification:
    //  - reachable addresses cannot be reached multiple ways
    //  - manager access windows fully map to downstream endpoints
    //  - burst kinds in manager windows are supported by the subordinates
    //    downstream
    //  - the network is loop-free
  }
};
} // namespace
