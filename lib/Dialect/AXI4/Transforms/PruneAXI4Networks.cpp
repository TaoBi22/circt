//===- PruneAXI4Networks.cpp - Prune AXI4 networks ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Removes the parts of an AXI4 network that no manager can address.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace axi4 {
#define GEN_PASS_DEF_PRUNEAXI4NETWORKS
#include "circt/Dialect/AXI4/AXI4Passes.h.inc"
} // namespace axi4
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;

/// The ops carrying `port` onwards, ending with the endpoint consuming it, or
/// failure if the connection reaches something this pass will not erase.
static FailureOr<SmallVector<Operation *>> collectBranch(Value port) {
  SmallVector<Operation *> branch;
  while (!port.use_empty()) {
    // A port has at most one use, so its consumer is the whole connection.
    Operation *consumer = *port.getUsers().begin();
    branch.push_back(consumer);

    // An adaptor carries the connection onwards
    if (isa<CutOp, CDCOp, DWConverterOp, IWConverterOp, BurstSplitterOp,
            BurstUnwrapperOp>(consumer)) {
      port = consumer->getResult(0);
      continue;
    }

    // An abstract subordinate ends it, and drives nothing else
    if (isa<AbstractSubordinateOp>(consumer))
      return branch;

    // A bridge out of the dialect ends it too, but only if the interface it
    // drives is unused - otherwise erasing it would strand live HW logic
    if (isa<PortToChannelStructsOp, ToMemOp>(consumer) &&
        llvm::all_of(consumer->getResults(),
                     [](Value result) { return result.use_empty(); }))
      return branch;

    return failure();
  }
  return branch;
}

/// Rebuild `op` without the downstream ports marked in `drop`.
template <typename Op>
static void dropDownstream(Op op, const llvm::SmallBitVector &drop) {
  OpBuilder builder(op);
  SmallVector<Type> types;
  SmallVector<Value> kept;
  for (auto [i, value] : llvm::enumerate(op.getDownstream())) {
    if (drop[i])
      continue;
    types.push_back(value.getType());
    kept.push_back(value);
  }

  auto rebuilt = Op::create(builder, op.getLoc(), types, op->getOperands(),
                            op->getAttrs());
  for (auto [before, after] : llvm::zip_equal(kept, rebuilt.getDownstream()))
    before.replaceAllUsesWith(after);
  op->erase();
}

/// Remove every downstream port of `op` that no upstream manager can address
/// and whose connection this pass can erase, warning about the rest.
template <typename Op>
static void pruneRouting(Op op, ValueRange upstream) {
  llvm::SmallBitVector drop(op.getDownstream().size());

  for (auto [i, value] : llvm::enumerate(op.getDownstream())) {
    if (isReachable(cast<PortType>(value.getType()), upstream))
      continue;

    FailureOr<SmallVector<Operation *>> branch = collectBranch(value);
    if (failed(branch)) {
      InFlightDiagnostic diag = op.emitWarning()
                                << "downstream port #" << i
                                << " is not addressed by any upstream manager";
      diag.attachNote((*value.getUsers().begin())->getLoc())
          << "connected to this operation, which the pass will not remove";
      continue;
    }

    // Erase from the endpoint back, so nothing is erased while still in use
    for (Operation *dead : llvm::reverse(*branch))
      dead->erase();
    drop[i] = true;
    op.emitRemark() << "removed downstream port #" << i
                    << ", which no upstream manager addresses";
  }

  if (drop.any())
    dropDownstream(op, drop);
}

namespace {
struct PruneAXI4NetworksPass
    : public circt::axi4::impl::PruneAXI4NetworksBase<PruneAXI4NetworksPass> {
  void runOnOperation() override;
};
} // namespace

void PruneAXI4NetworksPass::runOnOperation() {
  // Collect first, since pruning replaces the ops it walks
  SmallVector<Operation *> routing;
  getOperation()->walk([&](Operation *op) {
    if (isa<XbarOp, DemuxOp>(op))
      routing.push_back(op);
  });

  for (Operation *op : routing)
    TypeSwitch<Operation *>(op)
        .Case<XbarOp>(
            [](XbarOp xbar) { pruneRouting(xbar, xbar.getUpstream()); })
        .Case<DemuxOp>(
            [](DemuxOp demux) { pruneRouting(demux, demux.getUpstream()); });
}
