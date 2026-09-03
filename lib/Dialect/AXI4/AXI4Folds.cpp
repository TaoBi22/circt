//===- AXI4Folds.cpp - AXI4 canonicalizations -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Canonicalizations for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "mlir/IR/PatternMatch.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Erase an op whose upstream and downstream ports have the same type, so it
/// does nothing to the connection it sits on. A downstream port type describes
/// everything an op does to a connection, so equal types mean no change.
static LogicalResult eraseIdentity(Operation *op, Value upstream,
                                   Value downstream,
                                   PatternRewriter &rewriter) {
  if (upstream.getType() != downstream.getType())
    return rewriter.notifyMatchFailure(op, "op changes the port type");
  rewriter.replaceAllUsesWith(downstream, upstream);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Adaptor helpers
//===----------------------------------------------------------------------===//

/// Fuse an adaptor with the like adaptor directly upstream of it, leaving one
/// adaptor converting straight to the final port type. `fuses` decides whether
/// the composition is the same conversion, given the original, intermediate and
/// final port types.
template <typename Op>
static LogicalResult
fuseAdaptorPair(Op op, PatternRewriter &rewriter,
                llvm::function_ref<bool(PortType, PortType, PortType)> fuses) {
  Value upstream = op.getUpstream();
  auto prev = upstream.getDefiningOp<Op>();
  if (!prev)
    return rewriter.notifyMatchFailure(op, "upstream is not a like adaptor");

  if (!fuses(cast<PortType>(prev.getUpstream().getType()),
             cast<PortType>(upstream.getType()),
             cast<PortType>(op.getDownstream().getType())))
    return rewriter.notifyMatchFailure(op, "conversions do not compose");

  rewriter.modifyOpInPlace(
      op, [&] { op.getUpstreamMutable().assign(prev.getUpstream()); });
  // Nothing here is `Pure`, so the adaptor left behind will not be dropped for
  // us.
  rewriter.eraseOp(prev);
  return success();
}

//===----------------------------------------------------------------------===//
// Routing helpers
//===----------------------------------------------------------------------===//

/// Drop the downstream ports of a routing op that nothing consumes and that no
/// upstream manager can address.
template <typename Op>
static LogicalResult dropDeadDownstream(Op op, ValueRange upstream,
                                        PatternRewriter &rewriter) {
  SmallVector<Type> types;
  SmallVector<Value> kept;
  for (Value downstream : op.getDownstream()) {
    auto port = cast<PortType>(downstream.getType());
    if (downstream.use_empty() && !isReachable(port, upstream))
      continue;
    types.push_back(port);
    kept.push_back(downstream);
  }
  if (kept.size() == op.getDownstream().size())
    return rewriter.notifyMatchFailure(op, "every downstream port is live");

  auto rebuilt = Op::create(rewriter, op.getLoc(), types, op->getOperands(),
                            op->getAttrs());
  for (auto [before, after] : llvm::zip_equal(kept, rebuilt.getDownstream()))
    rewriter.replaceAllUsesWith(before, after);
  rewriter.eraseOp(op);
  return success();
}

/// Whether `port` tags transactions with the same ID widths as `reference`.
static bool sameIdWidths(PortType port, PortType reference) {
  return port.getWriteIdWidth() == reference.getWriteIdWidth() &&
         port.getReadIdWidth() == reference.getReadIdWidth();
}

//===----------------------------------------------------------------------===//
// CDCOp
//===----------------------------------------------------------------------===//

LogicalResult CDCOp::canonicalize(CDCOp op, PatternRewriter &rewriter) {
  // A crossing's port types match by construction, so the clocks are the whole
  // test for whether it crosses anything.
  if (op.getUpstreamClock() != op.getDownstreamClock())
    return rewriter.notifyMatchFailure(op, "crossing changes clock domain");
  return eraseIdentity(op, op.getUpstream(), op.getDownstream(), rewriter);
}

//===----------------------------------------------------------------------===//
// DWConverterOp
//===----------------------------------------------------------------------===//

LogicalResult DWConverterOp::canonicalize(DWConverterOp op,
                                          PatternRewriter &rewriter) {
  if (succeeded(
          eraseIdentity(op, op.getUpstream(), op.getDownstream(), rewriter)))
    return success();
  // A data-width converter pair can always be fused
  return fuseAdaptorPair(op, rewriter,
                         [](PortType, PortType, PortType) { return true; });
}

//===----------------------------------------------------------------------===//
// IWConverterOp
//===----------------------------------------------------------------------===//

LogicalResult IWConverterOp::canonicalize(IWConverterOp op,
                                          PatternRewriter &rewriter) {
  if (succeeded(
          eraseIdentity(op, op.getUpstream(), op.getDownstream(), rewriter)))
    return success();
  // Transactions must be ordered if ID width is narrowed - widening again will
  // not undo this, so we cannot safely fuse a pair of ID width converters if
  // the intermediate width is the lowest ID width in the chain
  auto keepsNarrowest = [](uint32_t original, uint32_t intermediate,
                           uint32_t downstream) {
    return intermediate >= std::min(original, downstream);
  };
  return fuseAdaptorPair(
      op, rewriter,
      [&](PortType original, PortType intermediate, PortType downstream) {
        return keepsNarrowest(original.getWriteIdWidth(),
                              intermediate.getWriteIdWidth(),
                              downstream.getWriteIdWidth()) &&
               keepsNarrowest(original.getReadIdWidth(),
                              intermediate.getReadIdWidth(),
                              downstream.getReadIdWidth());
      });
}

//===----------------------------------------------------------------------===//
// BurstSplitterOp
//===----------------------------------------------------------------------===//

LogicalResult BurstSplitterOp::canonicalize(BurstSplitterOp op,
                                            PatternRewriter &rewriter) {
  if (succeeded(
          eraseIdentity(op, op.getUpstream(), op.getDownstream(), rewriter)))
    return success();
  // Splitting twice is equivalent to splitting once
  return fuseAdaptorPair(op, rewriter,
                         [](PortType, PortType, PortType) { return true; });
}

//===----------------------------------------------------------------------===//
// BurstUnwrapperOp
//===----------------------------------------------------------------------===//

LogicalResult BurstUnwrapperOp::canonicalize(BurstUnwrapperOp op,
                                             PatternRewriter &rewriter) {
  if (succeeded(
          eraseIdentity(op, op.getUpstream(), op.getDownstream(), rewriter)))
    return success();
  // Unwrapping twice is equivalent to unwrapping once
  return fuseAdaptorPair(op, rewriter,
                         [](PortType, PortType, PortType) { return true; });
}

//===----------------------------------------------------------------------===//
// XbarOp
//===----------------------------------------------------------------------===//

LogicalResult XbarOp::canonicalize(XbarOp op, PatternRewriter &rewriter) {
  if (succeeded(dropDeadDownstream(op, op.getUpstream(), rewriter)))
    return success();

  ValueRange upstream = op.getUpstream();
  ValueRange downstream = op.getDownstream();

  // A crossbar between one manager and one subordinate that agree is a wire
  if (upstream.size() == 1 && downstream.size() == 1 &&
      succeeded(eraseIdentity(op, upstream[0], downstream[0], rewriter)))
    return success();

  // A single manager xbar has no transactions to tell apart, so it is a demux
  // (as long as it doesn't change ID widths)
  if (upstream.size() == 1) {
    auto upstreamTy = cast<PortType>(upstream[0].getType());
    if (llvm::all_of(downstream, [&](Value value) {
          return sameIdWidths(cast<PortType>(value.getType()), upstreamTy);
        })) {
      auto demux = DemuxOp::create(rewriter, op.getLoc(), downstream.getTypes(),
                                   op.getClock(), op.getReset(), upstream[0]);
      rewriter.replaceOp(op, demux.getDownstream());
      return success();
    }
  }

  // And an xbar with one subordinate can be treated as a mux
  if (downstream.size() == 1) {
    auto mux = MuxOp::create(rewriter, op.getLoc(), downstream[0].getType(),
                             op.getClock(), op.getReset(), upstream);
    rewriter.replaceOp(op, mux.getDownstream());
    return success();
  }

  return rewriter.notifyMatchFailure(op, "crossbar routes between ports");
}

//===----------------------------------------------------------------------===//
// DemuxOp
//===----------------------------------------------------------------------===//

LogicalResult DemuxOp::canonicalize(DemuxOp op, PatternRewriter &rewriter) {
  if (succeeded(dropDeadDownstream(op, op.getUpstream(), rewriter)))
    return success();
  if (op.getDownstream().size() != 1)
    return rewriter.notifyMatchFailure(op, "demux routes between ports");
  return eraseIdentity(op, op.getUpstream(), op.getDownstream()[0], rewriter);
}

//===----------------------------------------------------------------------===//
// MuxOp
//===----------------------------------------------------------------------===//

LogicalResult MuxOp::canonicalize(MuxOp op, PatternRewriter &rewriter) {
  if (op.getUpstream().size() != 1)
    return rewriter.notifyMatchFailure(op, "mux arbitrates between ports");
  return eraseIdentity(op, op.getUpstream()[0], op.getDownstream(), rewriter);
}
