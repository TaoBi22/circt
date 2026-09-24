//===- OptimizeAXI4Networks.cpp - Optimize AXI4 networks ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Optimizes AXI4 networks in ways that are not canonicalizations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace axi4 {
#define GEN_PASS_DEF_OPTIMIZEAXI4NETWORKS
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

//===----------------------------------------------------------------------===//
// Adaptor fusion
//===----------------------------------------------------------------------===//

/// The PULP config `prev` sets that `op` does not, or failure if the two set a
/// parameter to different values.
static FailureOr<SmallVector<NamedAttribute>>
pulpConfigToMerge(Operation *op, Operation *prev) {
  SmallVector<NamedAttribute> missing;
  for (NamedAttribute attr : prev->getDiscardableAttrs()) {
    if (!attr.getName().strref().starts_with(kPulpConfigPrefix))
      continue;
    Attribute existing = op->getDiscardableAttr(attr.getName());
    if (!existing)
      missing.push_back(attr);
    else if (existing != attr.getValue())
      return failure();
  }
  return missing;
}

/// Add `config`, taken from the op `op` is fused with, to `op`.
static void mergePulpConfig(Operation *op, ArrayRef<NamedAttribute> config,
                            PatternRewriter &rewriter) {
  if (config.empty())
    return;
  rewriter.modifyOpInPlace(op, [&] {
    for (NamedAttribute attr : config)
      op->setAttr(attr.getName(), attr.getValue());
  });
}

/// Search upstream from `port` for an adaptor of the same kind, stepping over
/// the cuts and crossings on the way and collecting them into `carriers`,
/// nearest `port` first. Null if anything else is reached first.
template <typename Op>
static Op findAdaptorThrough(Value port,
                             SmallVectorImpl<Operation *> &carriers) {
  while (Operation *def = port.getDefiningOp()) {
    if (auto adaptor = dyn_cast<Op>(def))
      return adaptor;
    if (!isa<CutOp, CDCOp>(def))
      break;
    carriers.push_back(def);
    port = TypeSwitch<Operation *, Value>(def).Case<CutOp, CDCOp>(
        [](auto carrier) { return carrier.getUpstream(); });
  }
  return {};
}

/// Fuse `op` with the like adaptor driving it, leaving one adaptor converting
/// straight to `op`'s downstream type. Unlike the canonicalization, this looks
/// through cuts and crossings, and asks nothing of the conversion beyond
/// composing - narrowing ID widths merges the IDs in flight, so a fused pair
/// can leave transactions free to complete in an order the original ordered.
template <typename Op>
static LogicalResult fuseAdaptors(Op op, PatternRewriter &rewriter) {
  SmallVector<Operation *> carriers;
  Op prev = findAdaptorThrough<Op>(op.getUpstream(), carriers);
  if (!prev)
    return rewriter.notifyMatchFailure(op, "no like adaptor upstream");
  FailureOr<SmallVector<NamedAttribute>> config = pulpConfigToMerge(op, prev);
  if (failed(config))
    return rewriter.notifyMatchFailure(
        op, "adaptors set a PULP parameter to different values");

  Value original = prev.getUpstream();
  if (carriers.empty()) {
    rewriter.modifyOpInPlace(op,
                             [&] { op.getUpstreamMutable().assign(original); });
  } else {
    // The carriers take their port unchanged, so they move onto the fused
    // adaptor's upstream and carry its type
    Operation *earliest = carriers.back();
    rewriter.modifyOpInPlace(earliest, [&] {
      TypeSwitch<Operation *>(earliest).Case<CutOp, CDCOp>(
          [&](auto carrier) { carrier.getUpstreamMutable().assign(original); });
    });
    for (Operation *carrier : carriers)
      rewriter.modifyOpInPlace(
          carrier, [&] { carrier->getResult(0).setType(original.getType()); });
  }
  mergePulpConfig(op, *config, rewriter);
  rewriter.eraseOp(prev);

  // A pair restoring the original port type fuses into an adaptor that does
  // nothing
  if (op.getUpstream().getType() == op.getDownstream().getType())
    rewriter.replaceOp(op, op.getUpstream());
  return success();
}

namespace {
/// Fuse every adaptor with the like adaptor driving it.
template <typename Op>
struct FuseAdaptors : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    return fuseAdaptors(op, rewriter);
  }
};

/// Fuse every crossing with the crossing driving it, so a connection crosses
/// once into the domain it ends up in.
struct FuseCrossings : OpRewritePattern<CDCOp> {
  using OpRewritePattern<CDCOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CDCOp op,
                                PatternRewriter &rewriter) const override {
    // Adjacent only: a cut in between would cross into the upstream domain
    // along with the port it carries
    auto prev = op.getUpstream().getDefiningOp<CDCOp>();
    if (!prev)
      return rewriter.notifyMatchFailure(op, "upstream is not a crossing");
    FailureOr<SmallVector<NamedAttribute>> config = pulpConfigToMerge(op, prev);
    if (failed(config))
      return rewriter.notifyMatchFailure(
          op, "crossings set a PULP parameter to different values");

    rewriter.modifyOpInPlace(op, [&] {
      op.getUpstreamMutable().assign(prev.getUpstream());
      op.getUpstreamClockMutable().assign(prev.getUpstreamClock());
    });
    mergePulpConfig(op, *config, rewriter);
    rewriter.eraseOp(prev);

    // A chain ending in the domain it started in crosses nothing
    if (op.getUpstreamClock() == op.getDownstreamClock())
      rewriter.replaceOp(op, op.getUpstream());
    return success();
  }
};

struct OptimizeAXI4NetworksPass
    : public circt::axi4::impl::OptimizeAXI4NetworksBase<
          OptimizeAXI4NetworksPass> {
  void runOnOperation() override;
};
} // namespace

void OptimizeAXI4NetworksPass::runOnOperation() {
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

  MLIRContext &context = getContext();
  RewritePatternSet patterns(&context);
  patterns.add<FuseAdaptors<DWConverterOp>, FuseAdaptors<IWConverterOp>,
               FuseAdaptors<BurstSplitterOp>, FuseAdaptors<BurstUnwrapperOp>,
               FuseCrossings>(&context);
  // Rewrite only the AXI4 ops, leaving the logic around them untouched
  Dialect *dialect = context.getLoadedDialect<AXI4Dialect>();
  SmallVector<Operation *> ops;
  getOperation()->walk([&](Operation *op) {
    if (op->getDialect() == dialect)
      ops.push_back(op);
  });
  GreedyRewriteConfig config;
  config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
  if (failed(applyOpPatternsGreedily(ops, std::move(patterns), config))) {
    getOperation().emitError("AXI4 network optimization did not converge");
    signalPassFailure();
  }
}
