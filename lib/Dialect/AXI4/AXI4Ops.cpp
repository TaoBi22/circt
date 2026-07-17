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
#include "llvm/ADT/StringMap.h"

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

/// Verify that `node` and `port_mapping` are either both given or both
/// omitted.
static LogicalResult verifyNodePortMapping(Operation *op, Value node,
                                           Attribute portMapping) {
  if (!node != !portMapping)
    return op->emitOpError(
        "'node' and 'port_mapping' must either both be given or both be "
        "omitted");
  return success();
}

//===----------------------------------------------------------------------===//
// NodeOp
//===----------------------------------------------------------------------===//

LogicalResult NodeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyModuleSymbol(*this, getModuleAttr(), symbolTable);
}

LogicalResult NodeOp::verify() {
  // Ports attached to this node that share a clock/reset port name in their
  // 'port_mapping' must also share the corresponding '!axi4.clock'/
  // '!axi4.reset' operand.
  llvm::StringMap<Value> clockOfPort, resetOfPort;
  auto checkShared = [&](llvm::StringMap<Value> &seen, StringRef port,
                         Value operand, StringRef kind) -> LogicalResult {
    auto [it, inserted] = seen.try_emplace(port, operand);
    if (!inserted && it->second != operand)
      return emitOpError("ports sharing the ")
             << kind << " port '" << port << "' must share the same '!axi4."
             << kind << "' operand";
    return success();
  };

  for (Operation *user : getNode().getUsers()) {
    AXI4PortMappingAttrInterface mapping;
    Value clock, reset;
    if (auto mgr = dyn_cast<ManagerPortOp>(user)) {
      mapping = mgr.getPortMappingAttr();
      clock = mgr.getClock();
      reset = mgr.getReset();
    } else if (auto sub = dyn_cast<SubordinatePortOp>(user)) {
      mapping = sub.getPortMappingAttr();
      clock = sub.getClock();
      reset = sub.getReset();
    } else {
      continue;
    }
    if (!mapping)
      continue;

    if (failed(
            checkShared(clockOfPort, mapping.getClockPort(), clock, "clock")))
      return failure();
    if (failed(
            checkShared(resetOfPort, mapping.getResetPort(), reset, "reset")))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ManagerPortOp
//===----------------------------------------------------------------------===//

LogicalResult ManagerPortOp::verify() {
  if (failed(verifyNodePortMapping(*this, getNode(), getPortMappingAttr())))
    return failure();
  if (failed(verifyAccessWindows(*this, getAccess())))
    return failure();
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
  if (failed(verifyNodePortMapping(*this, getNode(), getPortMappingAttr())))
    return failure();
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
    return op->emitError() << "xbar return type's " << name
                           << " must be at least the input " << name
                           << " + ceil(log2(number of managers)) (i.e., "
                           << minWidth << ")";
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
// DWConverterOp
//===----------------------------------------------------------------------===//

LogicalResult DWConverterOp::verify() {
  auto up = cast<PortType>(getUpstream().getType());
  auto down = cast<PortType>(getDownstream().getType());
  // A data width converter changes only the data width; the address, ID, and
  // user widths must be preserved.
  if (up.getAddressWidth() != down.getAddressWidth())
    return emitOpError("upstream and downstream address widths must match");
  if (up.getWriteIdWidth() != down.getWriteIdWidth() ||
      up.getReadIdWidth() != down.getReadIdWidth())
    return emitOpError("upstream and downstream ID widths must match");
  if (up.getUserWidth() != down.getUserWidth())
    return emitOpError("upstream and downstream user widths must match");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.cpp.inc"
