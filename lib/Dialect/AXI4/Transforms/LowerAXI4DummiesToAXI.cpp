//===- LowerAXI4DummiesToAXI.cpp - Lower the dummies subdialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers a network described in the dummies subdialect to the AXI4 dialect,
// inferring the parameterisation the dummies ops leave out.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
namespace axi4 {
#define GEN_PASS_DEF_LOWERAXI4DUMMIESTOAXI
#include "circt/Dialect/AXI4/AXI4Passes.h.inc"
} // namespace axi4
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Inference helpers
//===----------------------------------------------------------------------===//

/// The same bursts counted in beats of `to` bits rather than `from`. A burst
/// carries the same bytes either way, so its length scales with the width.
static FailureOr<BurstSetAttr> convertBursts(Operation *op, BurstSetAttr bursts,
                                             uint32_t from, uint32_t to) {
  if (from == to)
    return bursts;

  SmallVector<BurstSpecAttr> converted;
  for (BurstSpecAttr spec : bursts.getBurstSpecs()) {
    uint64_t bits = uint64_t{spec.getLen()} * from;
    if (bits % to)
      return op->emitOpError()
             << "burst " << spec << " does not divide into whole " << to
             << "-bit beats";
    BurstSpecAttr beats = BurstSpecAttr::getChecked(
        [&] {
          return op->emitOpError()
                 << "burst " << spec << " has no " << to << "-bit equivalent: ";
        },
        op->getContext(), spec.getKind(), static_cast<uint32_t>(bits / to));
    if (!beats)
      return failure();
    converted.push_back(beats);
  }
  return BurstSetAttr::get(op->getContext(), converted);
}

/// The windows a manager reaches, taken from the subordinates it declares
/// accesses to and the bursts it declares them with.
static FailureOr<WindowSetAttr>
inferWindows(DummiesExtManagerOp manager,
             ArrayRef<DummiesAccessesOp> accesses) {
  SmallVector<WindowAttr> windows;
  for (DummiesAccessesOp access : accesses) {
    auto subordinate =
        access.getSubordinate().getDefiningOp<DummiesExtSubordinateOp>();
    WindowAttr window = subordinate.getWindow();

    // The declared bursts are the manager's, so the subordinate supports them
    // in beats of its own data width.
    BurstSetAttr bursts = access.getBursts();
    FailureOr<BurstSetAttr> supported = convertBursts(
        access, bursts, manager.getDataWidth(), subordinate.getDataWidth());
    if (failed(supported))
      return failure();
    if (!window.getBurstSpecs().covers(*supported))
      return access.emitOpError()
             << "declares bursts " << bursts
             << " the subordinate does not support in " << window;
    windows.push_back(WindowAttr::get(access.getContext(), window.getBase(),
                                      window.getLast(), bursts));
  }

  if (windows.empty())
    return manager.emitOpError("must declare an access to reach a subordinate");
  return WindowSetAttr::get(manager.getContext(), windows);
}

/// Warn where a subordinate can hold fewer outstanding requests than the
/// manager reaching it can issue (this can't be caught post-lowering)
static void warnBottleneck(DummiesExtSubordinateOp subordinate, uint32_t writes,
                           uint32_t reads) {
  if (subordinate.getOutstandingWrites() < writes)
    subordinate.emitWarning()
        << "can hold fewer outstanding writes than the manager reaching it can "
           "issue ("
        << subordinate.getOutstandingWrites() << " < " << writes << ")";
  if (subordinate.getOutstandingReads() < reads)
    subordinate.emitWarning()
        << "can hold fewer outstanding reads than the manager reaching it can "
           "issue ("
        << subordinate.getOutstandingReads() << " < " << reads << ")";
}

/// Check a subordinate agrees with whatever reaches it on the widths neither a
/// crossbar nor a connection can change.
static LogicalResult checkSubordinate(DummiesExtSubordinateOp subordinate,
                                      const Twine &source, uint32_t addrWidth,
                                      uint32_t dataWidth, uint32_t writeIdWidth,
                                      uint32_t readIdWidth) {
  if (subordinate.getAddrWidth() != addrWidth)
    return subordinate.emitOpError()
           << "'addr_width' (" << subordinate.getAddrWidth() << ") must match "
           << "the " << source << "'s (" << addrWidth << ")";
  if (subordinate.getDataWidth() != dataWidth)
    return subordinate.emitOpError()
           << "'data_width' (" << subordinate.getDataWidth() << ") must match "
           << "the " << source << "'s (" << dataWidth
           << "); inserting data width converters is not yet implemented";
  if (llvm::Log2_64_Ceil(subordinate.getOutstandingWrites()) != writeIdWidth ||
      llvm::Log2_64_Ceil(subordinate.getOutstandingReads()) != readIdWidth)
    return subordinate.emitOpError()
           << "needs different ID widths to the " << source
           << " reaching it; inserting ID width converters is not yet "
              "implemented";
  return success();
}

/// Whether a type belongs to the dummies subdialect.
static bool isDummiesType(Type type) {
  return isa<DummiesPortType, DummiesManagerAccessType,
             DummiesSubordinateAccessType>(type);
}

/// The connections a dummies op consumes, in operand order.
static SmallVector<OpOperand *> incomingConnections(Operation *op) {
  SmallVector<OpOperand *> connections;
  for (OpOperand &operand : op->getOpOperands())
    if (isa<DummiesPortType>(operand.get().getType()))
      connections.push_back(&operand);
  return connections;
}

/// The connections a dummies value feeds, one per use.
static SmallVector<OpOperand *> outgoingConnections(Value port) {
  SmallVector<OpOperand *> connections;
  for (OpOperand &use : port.getUses())
    connections.push_back(&use);
  return connections;
}

//===----------------------------------------------------------------------===//
// Network lowering
//===----------------------------------------------------------------------===//

namespace {
/// Lowers the dummies network a module describes. Each connection - a use of a
/// dummies port - becomes one `!axi4.port` value. Windows are calculated by
/// propagating them up from subordinates, and widths are calculated by
/// propagating them down from managers

struct NetworkLowering {
  NetworkLowering(hw::HWModuleOp module) : module(module) {
    for (const hw::PortInfo &port : module.getPortList())
      names.newName(port.name.getValue());
  }

  /// Collect the network, and whether the module describes one at all.
  bool collect();
  LogicalResult lower(const DenseSet<StringAttr> &instantiated);

private:
  LogicalResult checkOneModule();
  LogicalResult inferManagerTypes();
  void emit();

  hw::HWModuleOp module;
  Namespace names;

  SmallVector<DummiesExtManagerOp> managers;
  SmallVector<DummiesExtSubordinateOp> subordinates;
  SmallVector<DummiesAccessesOp> accesses;
  SmallVector<DummiesXbarOp> xbars;

  /// The accesses each manager declares.
  DenseMap<Operation *, SmallVector<DummiesAccessesOp>> declared;
  /// The port type each connection carries.
  DenseMap<OpOperand *, PortType> types;
  /// The `!axi4.port` value feeding each connection.
  DenseMap<OpOperand *, Value> lowered;
};
} // namespace

bool NetworkLowering::collect() {
  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<DummiesExtManagerOp>([&](auto op) { managers.push_back(op); })
        .Case<DummiesExtSubordinateOp>(
            [&](auto op) { subordinates.push_back(op); })
        .Case<DummiesAccessesOp>([&](auto op) { accesses.push_back(op); })
        .Case<DummiesXbarOp>([&](auto op) { xbars.push_back(op); });
  });
  return !managers.empty() || !subordinates.empty() || !xbars.empty();
}

/// A network must be described in one module, so every dummies value in it
/// comes from a dummies op.
LogicalResult NetworkLowering::checkOneModule() {
  for (BlockArgument arg : module.getBodyBlock()->getArguments())
    if (isDummiesType(arg.getType()))
      return module.emitOpError(
          "cannot lower a dummies network reached through a module port; a "
          "network must be described in a single module");

  Dialect *axi4Dialect = module.getContext()->getLoadedDialect<AXI4Dialect>();
  WalkResult crossing = module.walk([&](Operation *op) {
    if (op->getDialect() == axi4Dialect)
      return WalkResult::advance();
    for (Value result : op->getResults())
      if (isDummiesType(result.getType())) {
        op->emitOpError("produces a dummies value; a network must be described "
                        "in a single module");
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return failure(crossing.wasInterrupted());
}

/// Give the connection each manager drives the port type the manager declares,
/// with the windows its accesses grant it.
LogicalResult NetworkLowering::inferManagerTypes() {
  for (DummiesExtManagerOp manager : managers) {
    SmallVector<OpOperand *> connections =
        outgoingConnections(manager.getPort());

    // Every access the manager declares must be to the subordinate it reaches.
    Operation *reached =
        connections.empty() ? nullptr : connections.front()->getOwner();
    for (DummiesAccessesOp access : declared[manager])
      if (access.getSubordinate().getDefiningOp() != reached)
        return access.emitOpError(
            "declares an access to a subordinate the manager cannot reach");

    // An access is what gives the manager its windows, so it drives a
    // connection from here on.
    FailureOr<WindowSetAttr> windows = inferWindows(manager, declared[manager]);
    if (failed(windows))
      return failure();

    // An endpoint needs enough ID bits to tag every request it can have
    // outstanding.
    uint32_t writeIdWidth = llvm::Log2_64_Ceil(manager.getOutstandingWrites());
    uint32_t readIdWidth = llvm::Log2_64_Ceil(manager.getOutstandingReads());
    types.insert({connections.front(),
                  PortType::get(module.getContext(), manager.getAddrWidth(),
                                manager.getDataWidth(), writeIdWidth,
                                readIdWidth, /*user_width=*/0, *windows,
                                manager.getOutstandingWrites(),
                                manager.getOutstandingReads())});

    // A subordinate the manager reaches without a crossbar shares this one
    // port type with it.
    if (auto subordinate =
            dyn_cast<DummiesExtSubordinateOp>(connections.front()->getOwner()))
      if (failed(checkSubordinate(
              subordinate, "manager", manager.getAddrWidth(),
              manager.getDataWidth(), writeIdWidth, readIdWidth)))
        return failure();
  }
  return success();
}

/// Replace the network with AXI4 ops, and its external endpoints with ports of
/// the module describing it.
void NetworkLowering::emit() {
  for (DummiesExtManagerOp manager : managers) {
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    auto [name, arg] =
        module.appendInput(names.newName(manager.getName().value_or("manager")),
                           types[connection]);
    lowered.insert({connection, arg});
  }

  for (DummiesExtSubordinateOp subordinate : subordinates) {
    OpOperand *connection = incomingConnections(subordinate).front();
    module.appendOutput(
        names.newName(subordinate.getName().value_or("subordinate")),
        lowered[connection]);
  }

  for (DummiesAccessesOp access : accesses)
    access.erase();
  for (DummiesExtSubordinateOp subordinate : subordinates)
    subordinate.erase();
  for (DummiesExtManagerOp manager : managers)
    manager.erase();
}

LogicalResult NetworkLowering::lower(const DenseSet<StringAttr> &instantiated) {
  if (!xbars.empty())
    return xbars.front().emitOpError(
        "lowering crossbars is not yet implemented");
  if (instantiated.contains(module.getModuleNameAttr()))
    return module.emitOpError(
        "cannot lower a dummies network in an instantiated module; its "
        "external endpoints must become ports of a top-level module");
  if (failed(checkOneModule()))
    return failure();

  for (DummiesAccessesOp access : accesses)
    declared[access.getManager().getDefiningOp()].push_back(access);

  if (failed(inferManagerTypes()))
    return failure();

  // A subordinate reached without a crossbar shares one port type with the
  // manager, so how many requests it can hold is only visible here.
  for (DummiesExtManagerOp manager : managers) {
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    if (auto subordinate =
            dyn_cast<DummiesExtSubordinateOp>(connection->getOwner()))
      warnBottleneck(subordinate, manager.getOutstandingWrites(),
                     manager.getOutstandingReads());
  }

  emit();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerAXI4DummiesToAXIPass
    : public circt::axi4::impl::LowerAXI4DummiesToAXIBase<
          LowerAXI4DummiesToAXIPass> {
  void runOnOperation() override;
};
} // namespace

void LowerAXI4DummiesToAXIPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Adding ports to a module would break any instance of it.
  DenseSet<StringAttr> instantiated;
  module.walk([&](hw::InstanceOp instance) {
    instantiated.insert(instance.getReferencedModuleNameAttr());
  });

  for (auto hwModule : module.getOps<hw::HWModuleOp>()) {
    NetworkLowering lowering(hwModule);
    if (!lowering.collect())
      continue;
    if (failed(lowering.lower(instantiated)))
      return signalPassFailure();
  }
}
