//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AXI4 ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Builders.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Verifier helpers
//===----------------------------------------------------------------------===//

/// Verify that `module` resolves to an `hw.module`-like symbol.
static LogicalResult verifyModuleSymbol(Operation *op, FlatSymbolRefAttr module,
                                        SymbolTableCollection &symbolTable) {
  Operation *moduleOp = symbolTable.lookupNearestSymbolFrom(op, module);
  if (!moduleOp)
    return op->emitOpError("references unknown symbol @") << module.getValue();
  if (!isa<hw::HWModuleLike>(moduleOp))
    return op->emitOpError("symbol @")
           << module.getValue() << " must refer to an 'hw.module'";
  return success();
}

/// Verify that no two access windows in `access` overlap.
static LogicalResult verifyAccessWindows(Operation *op, ArrayAttr access) {
  for (auto [i, lhsAttr] : llvm::enumerate(access)) {
    auto lhs = cast<WindowAttr>(lhsAttr);
    uint64_t lhsBase = lhs.getBase();
    uint64_t lhsEnd = lhsBase + lhs.getSize();
    for (auto rhsAttr : access.getValue().drop_front(i + 1)) {
      auto rhs = cast<WindowAttr>(rhsAttr);
      uint64_t rhsBase = rhs.getBase();
      uint64_t rhsEnd = rhsBase + rhs.getSize();
      if (lhsBase < rhsEnd && rhsBase < lhsEnd)
        return op->emitOpError("access windows overlap");
    }
  }
  return success();
}

/// Verify that `outstanding` does not exceed the 2^(ID width) outstanding
/// transactions addressable by an ID field of `idWidth` bits.
static LogicalResult verifyOutstanding(Operation *op, uint32_t idWidth,
                                       uint32_t outstanding, StringRef name) {
  // A `ui32` count can never exceed 2^32, so only widths below 32 can be
  // exceeded.
  if (idWidth < 32 && outstanding > (uint32_t(1) << idWidth))
    return op->emitOpError(name)
           << " (" << outstanding << ") exceeds the maximum of 2^" << idWidth
           << " (" << (uint32_t(1) << idWidth)
           << ") addressable by the port's ID width";
  return success();
}

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

LogicalResult NodeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyModuleSymbol(*this, getModuleAttr(), symbolTable);
}

//===----------------------------------------------------------------------===//
// ManagerPortOp
//===----------------------------------------------------------------------===//

LogicalResult ManagerPortOp::verify() {
  if (failed(verifyAccessWindows(*this, getAccess())))
    return failure();
  // Managers fanning out to multiple endpoints must go through an 'axi4.xbar'.
  if (getPort().hasNUsesOrMore(2))
    return emitOpError("result must have at most one use; route through an "
                       "'axi4.xbar' to fan out to multiple endpoints");
  auto port = cast<PortType>(getPort().getType());
  if (failed(verifyOutstanding(*this, port.getReadIdWidth(),
                               getOutstandingReads(), "outstanding_reads")))
    return failure();
  return verifyOutstanding(*this, port.getWriteIdWidth(),
                           getOutstandingWrites(), "outstanding_writes");
}

//===----------------------------------------------------------------------===//
// SubordinatePortOp
//===----------------------------------------------------------------------===//

LogicalResult SubordinatePortOp::verify() {
  return verifyAccessWindows(*this, getAccess());
}

//===----------------------------------------------------------------------===//
// XbarOp
//===----------------------------------------------------------------------===//

/// Verify that `downstreamWidth` is at least `upstreamWidth` +
/// ceil(log2(numManagers)), i.e. wide enough for the xbar to tag each
/// manager's transactions with a unique ID.
static LogicalResult verifyXbarIdWidth(Operation *op, uint32_t upstreamWidth,
                                       uint32_t downstreamWidth,
                                       size_t numManagers, StringRef name) {
  uint32_t minWidth = upstreamWidth + llvm::Log2_64_Ceil(numManagers);
  if (downstreamWidth < minWidth)
    return op->emitError()
           << "xbar return type's " << name << " must be at least the input "
           << name << " + ceil(log2(number of managers)) (i.e., " << minWidth
           << ")";
  return success();
}

LogicalResult XbarOp::verify() {
  auto upstream = getUpstream();
  auto firstPortTy = cast<PortType>(upstream.getTypes().front());
  for (Value v : upstream.drop_front())
    if (v.getType() != firstPortTy)
      return emitOpError("all upstream ports must have the same type");

  auto downstream = getResult().getType();
  if (failed(verifyXbarIdWidth(*this, firstPortTy.getWriteIdWidth(),
                               downstream.getWriteIdWidth(), upstream.size(),
                               "write id width")))
    return failure();
  return verifyXbarIdWidth(*this, firstPortTy.getReadIdWidth(),
                           downstream.getReadIdWidth(), upstream.size(),
                           "read id width");
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.cpp.inc"
