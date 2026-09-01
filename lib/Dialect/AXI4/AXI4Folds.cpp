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
// Adaptor helpers
//===----------------------------------------------------------------------===//

/// Erase an adaptor whose upstream and downstream ports have the same type, so
/// it does nothing to the connection it sits on. An adaptor's downstream port
/// type describes everything it does, so equal types mean no conversion.
template <typename Op>
static LogicalResult eraseIdentityAdaptor(Op op, PatternRewriter &rewriter) {
  if (op.getUpstream().getType() != op.getDownstream().getType())
    return rewriter.notifyMatchFailure(op, "adaptor changes the port type");
  rewriter.replaceOp(op, op.getUpstream());
  return success();
}

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
// CDCOp
//===----------------------------------------------------------------------===//

LogicalResult CDCOp::canonicalize(CDCOp op, PatternRewriter &rewriter) {
  // A crossing's port types match by construction, so the clocks are the whole
  // test for whether it crosses anything.
  if (op.getUpstreamClock() != op.getDownstreamClock())
    return rewriter.notifyMatchFailure(op, "crossing changes clock domain");
  return eraseIdentityAdaptor(op, rewriter);
}

//===----------------------------------------------------------------------===//
// DWConverterOp
//===----------------------------------------------------------------------===//

LogicalResult DWConverterOp::canonicalize(DWConverterOp op,
                                          PatternRewriter &rewriter) {
  if (succeeded(eraseIdentityAdaptor(op, rewriter)))
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
  if (succeeded(eraseIdentityAdaptor(op, rewriter)))
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
  if (succeeded(eraseIdentityAdaptor(op, rewriter)))
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
  if (succeeded(eraseIdentityAdaptor(op, rewriter)))
    return success();
  // Unwrapping twice is equivalent to unwrapping once
  return fuseAdaptorPair(op, rewriter,
                         [](PortType, PortType, PortType) { return true; });
}
