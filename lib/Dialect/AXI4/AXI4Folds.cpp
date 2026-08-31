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
  return eraseIdentityAdaptor(op, rewriter);
}

//===----------------------------------------------------------------------===//
// IWConverterOp
//===----------------------------------------------------------------------===//

LogicalResult IWConverterOp::canonicalize(IWConverterOp op,
                                          PatternRewriter &rewriter) {
  return eraseIdentityAdaptor(op, rewriter);
}

//===----------------------------------------------------------------------===//
// BurstSplitterOp
//===----------------------------------------------------------------------===//

LogicalResult BurstSplitterOp::canonicalize(BurstSplitterOp op,
                                            PatternRewriter &rewriter) {
  return eraseIdentityAdaptor(op, rewriter);
}

//===----------------------------------------------------------------------===//
// BurstUnwrapperOp
//===----------------------------------------------------------------------===//

LogicalResult BurstUnwrapperOp::canonicalize(BurstUnwrapperOp op,
                                             PatternRewriter &rewriter) {
  return eraseIdentityAdaptor(op, rewriter);
}
