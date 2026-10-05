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
#include "llvm/ADT/BitVector.h"
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

/// The same windows with their bursts adjusted to a new data width
static FailureOr<WindowSetAttr> convertWindows(Operation *op,
                                               WindowSetAttr windows,
                                               uint32_t from, uint32_t to) {
  if (from == to)
    return windows;

  SmallVector<WindowAttr> converted;
  for (WindowAttr window : windows.getWindows()) {
    FailureOr<BurstSetAttr> bursts =
        convertBursts(op, window.getBurstSpecs(), from, to);
    if (failed(bursts))
      return failure();
    converted.push_back(WindowAttr::get(op->getContext(), window.getBase(),
                                        window.getLast(), *bursts));
  }
  return WindowSetAttr::get(op->getContext(), converted);
}

/// Converts a set of supported bursts to a new data_width,
/// rounding down to the nearest legal burst length (dropping specs for which
/// there is no legal length to round down to). Fails if there are no legal
/// bursts left after conversion.
static FailureOr<BurstSetAttr>
convertSupport(Operation *op, BurstSetAttr bursts, uint32_t from, uint32_t to) {
  if (from == to)
    return bursts;

  SmallVector<BurstSpecAttr> supported;
  for (BurstSpecAttr spec : bursts.getBurstSpecs()) {
    // A `len` is a maximum, so asking for a shorter burst is always safe.
    uint64_t beats = std::min<uint64_t>(uint64_t{spec.getLen()} * from / to,
                                        getMaxBurstLen(spec.getKind()));
    // Keep it a whole number of `from`-bit beats.
    if (to < from)
      beats -= beats % (from / to);
    if (beats == 0 || (spec.getKind() == BurstKind::Wrap && beats < 2))
      continue;
    supported.push_back(BurstSpecAttr::get(op->getContext(), spec.getKind(),
                                           static_cast<uint32_t>(beats)));
  }

  if (supported.empty())
    return op->emitOpError()
           << "supports no burst a port of " << to << " bits can ask for";
  return BurstSetAttr::get(op->getContext(), supported);
}

/// Check `served` covers every address the access declares, and supports
/// `supported` wherever it does.
static LogicalResult checkServed(DummiesAccessesOp access, WindowSetAttr served,
                                 BurstSetAttr supported) {
  WindowAttr window = access.getWindow();

  // The served windows are disjoint and sorted, so walking them in order
  // either reaches past the declared window or finds a gap in it.
  uint64_t next = window.getBase();
  for (WindowAttr candidate : served.getWindows()) {
    if (candidate.getLast() < next)
      continue;
    if (candidate.getBase() > next)
      break;
    if (!candidate.getBurstSpecs().covers(supported))
      return access.emitOpError()
             << "declares bursts " << window.getBurstSpecs()
             << " the subordinate does not support in " << candidate;
    if (candidate.getLast() >= window.getLast())
      return success();
    next = candidate.getLast() + 1;
  }
  return access.emitOpError() << "declares an access to " << window
                              << " the subordinate does not serve in full";
}

/// The windows a manager reaches, taken from the accesses it declares. Where
/// they overlap it reaches the union of the bursts they declare.
static FailureOr<WindowSetAttr>
inferWindows(DummiesExtManagerOp manager,
             ArrayRef<DummiesAccessesOp> accesses) {
  SmallVector<WindowAttr> windows;
  for (DummiesAccessesOp access : accesses) {
    auto subordinate =
        access.getSubordinate().getDefiningOp<DummiesExtSubordinateOp>();
    WindowAttr window = access.getWindow();

    // The declared bursts are the manager's, so the subordinate supports them
    // in beats of its own data width.
    FailureOr<BurstSetAttr> supported =
        convertBursts(access, window.getBurstSpecs(), manager.getDataWidth(),
                      subordinate.getDataWidth());
    if (failed(supported))
      return failure();
    if (failed(checkServed(access, subordinate.getWindows(), *supported)))
      return failure();
    windows.push_back(window);
  }

  if (windows.empty())
    return manager.emitOpError("must declare an access to reach a subordinate");
  return WindowSetAttr::get(manager.getContext(), windows);
}

/// Warn where a subordinate can hold fewer outstanding requests than the
/// managers reaching it can issue (this can't be caught post-lowering)
static void warnBottleneck(DummiesExtSubordinateOp subordinate, uint32_t writes,
                           uint32_t reads) {
  if (subordinate.getOutstandingWrites() < writes)
    subordinate.emitWarning()
        << "can hold fewer outstanding writes than the managers reaching it "
           "can issue ("
        << subordinate.getOutstandingWrites() << " < " << writes << ")";
  if (subordinate.getOutstandingReads() < reads)
    subordinate.emitWarning()
        << "can hold fewer outstanding reads than the managers reaching it can "
           "issue ("
        << subordinate.getOutstandingReads() << " < " << reads << ")";
}

/// Check a subordinate agrees with whatever reaches it on its address width,
/// which no converter currently changes.
static LogicalResult checkSubordinate(DummiesExtSubordinateOp subordinate,
                                      const Twine &source, uint32_t addrWidth) {
  if (subordinate.getAddrWidth() != addrWidth)
    return subordinate.emitOpError()
           << "'addr_width' (" << subordinate.getAddrWidth() << ") must match "
           << "the " << source << "'s (" << addrWidth << ")";
  return success();
}

/// As many of `outstanding` requests as `idWidth` ID bits can tag.
static uint32_t taggableOutstanding(uint32_t outstanding, uint32_t idWidth) {
  return static_cast<uint32_t>(
      std::min<uint64_t>(outstanding, uint64_t{1} << idWidth));
}

/// The port type a subordinate presents over `windows`, carrying the `writes`
/// and `reads` reaching it, with ID widths wide enough to tag every request it
/// can hold.
static PortType getSubordinatePortType(DummiesExtSubordinateOp subordinate,
                                       uint32_t userWidth,
                                       WindowSetAttr windows, uint32_t writes,
                                       uint32_t reads) {
  uint32_t writeIdWidth =
      llvm::Log2_64_Ceil(subordinate.getOutstandingWrites());
  uint32_t readIdWidth = llvm::Log2_64_Ceil(subordinate.getOutstandingReads());
  return PortType::get(subordinate.getContext(), subordinate.getAddrWidth(),
                       subordinate.getDataWidth(), writeIdWidth, readIdWidth,
                       userWidth, windows,
                       taggableOutstanding(writes, writeIdWidth),
                       taggableOutstanding(reads, readIdWidth));
}

/// The same port type with different ID widths.
static PortType getPortTypeWithIdWidths(PortType port, uint32_t writeIdWidth,
                                        uint32_t readIdWidth) {
  return PortType::get(port.getContext(), port.getAddrWidth(),
                       port.getDataWidth(), writeIdWidth, readIdWidth,
                       port.getUserWidth(), port.getWindows(),
                       port.getOutstandingWrites(), port.getOutstandingReads());
}

/// The same port type with a different data width, which its bursts are
/// counted in beats of.
static FailureOr<PortType>
getPortTypeWithDataWidth(Operation *op, PortType port, uint32_t dataWidth) {
  FailureOr<WindowSetAttr> windows =
      convertWindows(op, port.getWindows(), port.getDataWidth(), dataWidth);
  if (failed(windows))
    return failure();
  return PortType::get(port.getContext(), port.getAddrWidth(), dataWidth,
                       port.getWriteIdWidth(), port.getReadIdWidth(),
                       port.getUserWidth(), *windows,
                       port.getOutstandingWrites(), port.getOutstandingReads());
}

/// Whether a converter has to bridge a port to a connection carrying these
/// widths.
static bool needsConverter(PortType port, uint32_t dataWidth,
                           uint32_t writeIdWidth, uint32_t readIdWidth) {
  return port.getDataWidth() != dataWidth ||
         port.getWriteIdWidth() != writeIdWidth ||
         port.getReadIdWidth() != readIdWidth;
}

/// The clock and reset a dummies op runs on.
static std::pair<Value, Value> domainOf(Operation *op) {
  return TypeSwitch<Operation *, std::pair<Value, Value>>(op)
      .Case<DummiesExtManagerOp, DummiesExtSubordinateOp, DummiesXbarOp,
            DummiesCutOp, DummiesIDRemapOp>(
          [](auto op) { return std::make_pair(op.getClock(), op.getReset()); });
}

/// Check an endpoint runs in the domain of the op it connects to. It becomes a
/// module port, which carries no clock or reset of its own.
static LogicalResult checkDomain(Operation *endpoint, Operation *other) {
  auto [clock, reset] = domainOf(endpoint);
  auto [otherClock, otherReset] = domainOf(other);
  StringRef domain;
  if (clock != otherClock)
    domain = "clock";
  else if (reset != otherReset)
    domain = "reset";
  else
    return success();
  auto diag = endpoint->emitOpError()
              << "is in a different " << domain << " domain to the '"
              << other->getName().getStringRef() << "' connected to it";
  diag.attachNote(other->getLoc()) << "connected operation here";
  return diag;
}

/// The data width the op consuming a connection presents.
static uint32_t dataWidthOf(Operation *op) {
  if (auto xbar = dyn_cast<DummiesXbarOp>(op))
    return xbar.getDataWidth();
  return cast<DummiesExtSubordinateOp>(op).getDataWidth();
}

/// A connectivity matrix as a matrix of booleans, one row per upstream port.
static ArrayAttr getConnectivityAttr(MLIRContext *context,
                                     ArrayRef<llvm::BitVector> rows) {
  SmallVector<Attribute> matrix;
  for (const llvm::BitVector &row : rows) {
    SmallVector<Attribute> bits;
    for (unsigned bit = 0; bit != row.size(); ++bit)
      bits.push_back(BoolAttr::get(context, row[bit]));
    matrix.push_back(ArrayAttr::get(context, bits));
  }
  return ArrayAttr::get(context, matrix);
}

/// The PULP atop markers `endpoint` carries, as the attributes of the port it
/// becomes.
static DictionaryAttr getAtopMarkers(Operation *endpoint) {
  SmallVector<NamedAttribute> markers;
  for (StringRef name : {kPulpAtopsAttr, kPulpAtopFilterAttr})
    if (Attribute attr = endpoint->getAttr(name))
      markers.push_back(
          NamedAttribute(StringAttr::get(endpoint->getContext(), name), attr));
  return DictionaryAttr::get(endpoint->getContext(), markers);
}

/// Give port `index` of `module` the attributes `attrs`. The module's other
/// ports get empty ones if they had none, which the HW interface's
/// `setPortAttrs` leaves null.
static void setPortAttrs(hw::HWModuleOp module, size_t index,
                         DictionaryAttr attrs) {
  SmallVector<Attribute> all(module.getAllPortAttrs());
  all.resize(module.getNumPorts());
  for (Attribute &attr : all)
    if (!attr)
      attr = DictionaryAttr::get(module.getContext());
  all[index] = attrs;
  module.setAllPortAttrs(all);
}

/// The IDs a remapper tracks, of those its `upstream` port can carry.
static uint32_t trackedIds(DummiesIDRemapOp remap, PortType upstream) {
  uint32_t idWidth =
      std::min(upstream.getWriteIdWidth(), upstream.getReadIdWidth());
  return static_cast<uint32_t>(
      std::min<uint64_t>(remap.getMaxUniqueIds(), uint64_t{1} << idWidth));
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

/// The port a crossbar or remapper drives downstream.
static Value downstreamOf(Operation *stage) {
  return TypeSwitch<Operation *, Value>(stage)
      .Case<DummiesXbarOp, DummiesIDRemapOp>(
          [](auto op) { return op.getDownstream(); });
}

/// The use a connection ends at, past any cuts on it.
static OpOperand *throughCuts(OpOperand *connection) {
  while (auto cut = dyn_cast<DummiesCutOp>(connection->getOwner()))
    connection = &*cut.getDownstream().use_begin();
  return connection;
}

/// The connections a dummies value feeds, one per use.
static SmallVector<OpOperand *> outgoingConnections(Value port) {
  SmallVector<OpOperand *> connections;
  for (OpOperand &use : port.getUses())
    connections.push_back(throughCuts(&use));
  return connections;
}

/// Whether a cut is fed only through other cuts, around a cycle.
static bool isOnCutCycle(DummiesCutOp cut) {
  SmallPtrSet<Operation *, 8> seen;
  for (; cut; cut = cut.getUpstream().getDefiningOp<DummiesCutOp>())
    if (!seen.insert(cut).second)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// Network lowering
//===----------------------------------------------------------------------===//

namespace {
/// Lowers the dummies network a module describes. Each connection - a use of a
/// dummies port, followed through any cuts - carries one `!axi4.port` type.
/// Windows are calculated by propagating them up from subordinates, and widths
/// are calculated by propagating them down from managers

struct NetworkLowering {
  NetworkLowering(hw::HWModuleOp module, uint32_t userWidth)
      : module(module), userWidth(userWidth) {
    for (const hw::PortInfo &port : module.getPortList())
      names.newName(port.name.getValue());
  }

  /// Collect the network, and whether the module describes one at all.
  bool collect();
  LogicalResult lower(const DenseSet<StringAttr> &instantiated);

private:
  LogicalResult checkOneModule();
  FailureOr<SmallVector<DummiesExtSubordinateOp>>
  getReachableSubordinates(OpOperand *connection);
  FailureOr<WindowSetAttr> windowsBelow(OpOperand *connection,
                                        uint32_t dataWidth);
  LogicalResult adaptToSubordinate(OpOperand *connection, PortType driven,
                                   const Twine &source);
  LogicalResult inferManagerTypes();
  LogicalResult inferXbarTypes(DummiesXbarOp xbar);
  LogicalResult inferRemapTypes(DummiesIDRemapOp remap);
  LogicalResult inferTypes();
  LogicalResult inferConnectivity();
  void drive(OpOperand *connection, Value port);
  void emit();

  hw::HWModuleOp module;
  uint32_t userWidth;
  Namespace names;

  SmallVector<DummiesExtManagerOp> managers;
  SmallVector<DummiesExtSubordinateOp> subordinates;
  SmallVector<DummiesAccessesOp> accesses;
  SmallVector<DummiesXbarOp> xbars;
  SmallVector<DummiesCutOp> cuts;
  SmallVector<DummiesIDRemapOp> remaps;

  /// The accesses each manager declares.
  DenseMap<Operation *, SmallVector<DummiesAccessesOp>> declared;
  /// The subordinates each connection reaches, and the connections being
  /// visited.
  DenseMap<OpOperand *, SmallVector<DummiesExtSubordinateOp>>
      reachableSubordinates;
  DenseSet<OpOperand *> visiting;
  /// The port type each connection carries, and the one its consumer needs
  /// where a converter has to bridge the two.
  DenseMap<OpOperand *, PortType> types;
  DenseMap<OpOperand *, PortType> adapted;
  /// Which downstream ports each of a crossbar's upstream ports reaches.
  DenseMap<Operation *, SmallVector<llvm::BitVector>> connectivity;
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
        .Case<DummiesXbarOp>([&](auto op) { xbars.push_back(op); })
        .Case<DummiesCutOp>([&](auto op) { cuts.push_back(op); })
        .Case<DummiesIDRemapOp>([&](auto op) { remaps.push_back(op); });
  });
  return !managers.empty() || !subordinates.empty() || !xbars.empty() ||
         !cuts.empty() || !remaps.empty();
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

/// The subordinates a connection reaches, following the crossbars and
/// remappers below it.
FailureOr<SmallVector<DummiesExtSubordinateOp>>
NetworkLowering::getReachableSubordinates(OpOperand *connection) {
  if (auto it = reachableSubordinates.find(connection);
      it != reachableSubordinates.end())
    return it->second;
  if (!visiting.insert(connection).second)
    return connection->getOwner()->emitOpError(
        "is part of a cycle in the dummies network");

  SmallVector<DummiesExtSubordinateOp> found;
  if (auto subordinate =
          dyn_cast<DummiesExtSubordinateOp>(connection->getOwner())) {
    found.push_back(subordinate);
  } else {
    for (OpOperand *below :
         outgoingConnections(downstreamOf(connection->getOwner()))) {
      FailureOr<SmallVector<DummiesExtSubordinateOp>> reached =
          getReachableSubordinates(below);
      if (failed(reached))
        return failure();
      llvm::append_range(found, *reached);
    }
  }

  visiting.erase(connection);
  reachableSubordinates.insert({connection, found});
  return found;
}

/// The windows a connection carries: those of the subordinates below it, in
/// beats of `dataWidth`.
FailureOr<WindowSetAttr> NetworkLowering::windowsBelow(OpOperand *connection,
                                                       uint32_t dataWidth) {
  Operation *consumer = connection->getOwner();

  // A remapper leaves the beats alone.
  if (auto remap = dyn_cast<DummiesIDRemapOp>(consumer))
    return windowsBelow(outgoingConnections(remap.getDownstream()).front(),
                        dataWidth);

  // A crossbar carries the windows of everything below it, each of them
  // already in beats of its own data width.
  SmallVector<WindowAttr> presented;
  if (auto subordinate = dyn_cast<DummiesExtSubordinateOp>(consumer)) {
    llvm::append_range(presented, subordinate.getWindows().getWindows());
  } else {
    auto xbar = cast<DummiesXbarOp>(consumer);
    for (OpOperand *below : outgoingConnections(xbar.getDownstream())) {
      FailureOr<WindowSetAttr> windows =
          windowsBelow(below, xbar.getDataWidth());
      if (failed(windows))
        return failure();
      llvm::append_range(presented, windows->getWindows());
    }
  }

  SmallVector<WindowAttr> windows;
  for (WindowAttr window : presented) {
    FailureOr<BurstSetAttr> bursts = convertSupport(
        consumer, window.getBurstSpecs(), dataWidthOf(consumer), dataWidth);
    if (failed(bursts))
      return failure();
    windows.push_back(WindowAttr::get(module.getContext(), window.getBase(),
                                      window.getLast(), *bursts));
  }
  return WindowSetAttr::get(module.getContext(), windows);
}

/// Adapt a connection carrying `driven` straight to the subordinate consuming
/// it, if one does. The subordinate presents a port of its own: its own data
/// width, and its own ID widths, log2 of the requests it can hold.
LogicalResult NetworkLowering::adaptToSubordinate(OpOperand *connection,
                                                  PortType driven,
                                                  const Twine &source) {
  auto subordinate = dyn_cast<DummiesExtSubordinateOp>(connection->getOwner());
  if (!subordinate)
    return success();
  if (failed(checkSubordinate(subordinate, source, driven.getAddrWidth())))
    return failure();

  // It serves the bursts driven in beats of its own data width.
  FailureOr<WindowSetAttr> served =
      convertWindows(subordinate, driven.getWindows(), driven.getDataWidth(),
                     subordinate.getDataWidth());
  if (failed(served))
    return failure();
  PortType port = getSubordinatePortType(subordinate, userWidth, *served,
                                         driven.getOutstandingWrites(),
                                         driven.getOutstandingReads());
  if (needsConverter(port, driven.getDataWidth(), driven.getWriteIdWidth(),
                     driven.getReadIdWidth()))
    adapted.insert({connection, port});
  return success();
}

/// Give the connection each manager drives the port type the manager declares,
/// with the windows its accesses grant it.
LogicalResult NetworkLowering::inferManagerTypes() {
  for (DummiesExtManagerOp manager : managers) {
    SmallVector<OpOperand *> connections =
        outgoingConnections(manager.getPort());

    // Every access the manager declares must be to a subordinate it reaches.
    SmallVector<DummiesExtSubordinateOp> reached;
    if (!connections.empty()) {
      FailureOr<SmallVector<DummiesExtSubordinateOp>> below =
          getReachableSubordinates(connections.front());
      if (failed(below))
        return failure();
      reached = *below;
    }
    for (DummiesAccessesOp access : declared[manager])
      if (!llvm::is_contained(reached, access.getSubordinate().getDefiningOp()))
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
    types.insert(
        {connections.front(),
         PortType::get(module.getContext(), manager.getAddrWidth(),
                       manager.getDataWidth(), writeIdWidth, readIdWidth,
                       userWidth, *windows, manager.getOutstandingWrites(),
                       manager.getOutstandingReads())});

    if (failed(adaptToSubordinate(connections.front(),
                                  types[connections.front()], "manager")))
      return failure();
  }
  return success();
}

/// Give each of a crossbar's downstream connections a port type, from the types
/// of its upstream connections.
LogicalResult NetworkLowering::inferXbarTypes(DummiesXbarOp xbar) {
  SmallVector<OpOperand *> upstream = incomingConnections(xbar);
  SmallVector<OpOperand *> downstream =
      outgoingConnections(xbar.getDownstream());

  // A crossbar routes, it does not re-address, so it must agree with
  // everything it connects on the address width.
  uint32_t writeIdWidth = 0, readIdWidth = 0;
  for (OpOperand *connection : upstream) {
    PortType type = types[connection];
    if (type.getAddrWidth() != xbar.getAddrWidth())
      return xbar.emitOpError() << "'addr_width' (" << xbar.getAddrWidth()
                                << ") must match that of the port reaching it ("
                                << type.getAddrWidth() << ")";

    // Its upstream ports must all carry the same ID widths, so they are as
    // wide as the widest port reaching it.
    writeIdWidth = std::max(writeIdWidth, type.getWriteIdWidth());
    readIdWidth = std::max(readIdWidth, type.getReadIdWidth());
  }
  for (OpOperand *connection : upstream) {
    PortType type = types[connection];
    if (!needsConverter(type, xbar.getDataWidth(), writeIdWidth, readIdWidth))
      continue;
    FailureOr<PortType> needed = getPortTypeWithDataWidth(
        xbar, getPortTypeWithIdWidths(type, writeIdWidth, readIdWidth),
        xbar.getDataWidth());
    if (failed(needed))
      return failure();
    adapted.insert({connection, *needed});
  }

  // Transactions are tagged with the index of the manager they came from, so
  // the downstream ports carry wider IDs than the upstream ones.
  uint32_t tagBits = llvm::Log2_64_Ceil(upstream.size());
  writeIdWidth += tagBits;
  readIdWidth += tagBits;

  for (OpOperand *connection : downstream) {
    FailureOr<WindowSetAttr> windows =
        windowsBelow(connection, xbar.getDataWidth());
    if (failed(windows))
      return failure();

    // A connection carries what the managers that can address it issue there.
    uint32_t writes = 0, reads = 0;
    for (OpOperand *above : upstream) {
      PortType type = types[above];
      if (type.getWindows().overlaps(*windows)) {
        writes += type.getOutstandingWrites();
        reads += type.getOutstandingReads();
      }
    }

    PortType port;
    if (auto subordinate =
            dyn_cast<DummiesExtSubordinateOp>(connection->getOwner())) {
      if (failed(
              checkSubordinate(subordinate, "crossbar", xbar.getAddrWidth())))
        return failure();
      warnBottleneck(subordinate, writes, reads);

      // It serves the crossbar's bursts in beats of its own data width.
      FailureOr<WindowSetAttr> served =
          convertWindows(subordinate, *windows, xbar.getDataWidth(),
                         subordinate.getDataWidth());
      if (failed(served))
        return failure();
      port = getSubordinatePortType(subordinate, userWidth, *served, writes,
                                    reads);
    }

    // The crossbar tags requests with more ID bits than its managers use, so
    // what it drives is only what a subordinate presents if they happen to
    // agree.
    if (port &&
        needsConverter(port, xbar.getDataWidth(), writeIdWidth, readIdWidth))
      adapted.insert({connection, port});

    // The address widths below are only checked once their crossbars are
    // lowered, so the windows they present can be too wide for this one.
    auto emitError = [&]() { return xbar.emitOpError(); };
    if (failed(verifyWindowsFit(emitError, "", xbar.getAddrWidth(),
                                windows->getWindows())))
      return failure();

    types.insert({connection,
                  PortType::get(module.getContext(), xbar.getAddrWidth(),
                                xbar.getDataWidth(), writeIdWidth, readIdWidth,
                                userWidth, *windows, writes, reads)});
  }

  return success();
}

/// Give a remapper's downstream connection a port type, from the type of its
/// upstream connection: IDs just wide enough for the IDs it tracks, and as
/// many requests in flight as it tracks IDs.
LogicalResult NetworkLowering::inferRemapTypes(DummiesIDRemapOp remap) {
  PortType upstream = types[incomingConnections(remap).front()];
  OpOperand *connection = outgoingConnections(remap.getDownstream()).front();
  uint32_t uniqueIds = trackedIds(remap, upstream);
  uint32_t idWidth = llvm::Log2_64_Ceil(uniqueIds);
  uint32_t writes = std::min(upstream.getOutstandingWrites(), uniqueIds);
  uint32_t reads = std::min(upstream.getOutstandingReads(), uniqueIds);
  PortType driven = PortType::get(
      module.getContext(), upstream.getAddrWidth(), upstream.getDataWidth(),
      idWidth, idWidth, userWidth, upstream.getWindows(), writes, reads);
  types.insert({connection, driven});

  if (auto subordinate =
          dyn_cast<DummiesExtSubordinateOp>(connection->getOwner()))
    warnBottleneck(subordinate, writes, reads);
  if (failed(adaptToSubordinate(connection, driven, "remapper")))
    return failure();

  return success();
}

/// Give every connection in the network a port type, working down from the
/// managers.
LogicalResult NetworkLowering::inferTypes() {
  if (failed(inferManagerTypes()))
    return failure();

  // A crossbar or remapper can be lowered once everything reaching it has
  // been.
  SmallVector<Operation *> pending(xbars.begin(), xbars.end());
  pending.append(remaps.begin(), remaps.end());
  while (!pending.empty()) {
    SmallVector<Operation *> waiting;
    for (Operation *stage : pending) {
      if (!llvm::all_of(incomingConnections(stage), [&](OpOperand *connection) {
            return types.contains(connection);
          })) {
        waiting.push_back(stage);
        continue;
      }
      LogicalResult inferred =
          isa<DummiesXbarOp>(stage)
              ? inferXbarTypes(cast<DummiesXbarOp>(stage))
              : inferRemapTypes(cast<DummiesIDRemapOp>(stage));
      if (failed(inferred))
        return failure();
    }
    if (waiting.size() == pending.size())
      return waiting.front()->emitOpError(
          "is part of a cycle in the dummies network");
    pending = waiting;
  }
  return success();
}

/// Connect each crossbar's upstream port to the downstream ports on the way to
/// the subordinates accessed through it.
LogicalResult NetworkLowering::inferConnectivity() {
  for (DummiesXbarOp xbar : xbars)
    connectivity[xbar].assign(
        xbar.getUpstream().size(),
        llvm::BitVector(outgoingConnections(xbar.getDownstream()).size()));

  for (DummiesAccessesOp access : accesses) {
    Operation *subordinate = access.getSubordinate().getDefiningOp();
    auto manager = access.getManager().getDefiningOp<DummiesExtManagerOp>();
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    while (!isa<DummiesExtSubordinateOp>(connection->getOwner())) {
      if (auto remap = dyn_cast<DummiesIDRemapOp>(connection->getOwner())) {
        connection = outgoingConnections(remap.getDownstream()).front();
        continue;
      }
      auto xbar = cast<DummiesXbarOp>(connection->getOwner());
      unsigned upstream = connection->getOperandNumber() -
                          xbar.getUpstream().getBeginOperandIndex();
      for (auto [index, below] :
           llvm::enumerate(outgoingConnections(xbar.getDownstream()))) {
        FailureOr<SmallVector<DummiesExtSubordinateOp>> reached =
            getReachableSubordinates(below);
        if (failed(reached))
          return failure();
        if (llvm::is_contained(*reached, subordinate)) {
          connectivity[xbar][upstream].set(index);
          connection = below;
          break;
        }
      }
    }
  }
  return success();
}

/// Record the value a connection carries, through the cuts on it, converting
/// the widths of what its producer drives to what its consumer needs.
void NetworkLowering::drive(OpOperand *connection, Value port) {
  SmallVector<DummiesCutOp> onPath;
  for (auto cut = connection->get().getDefiningOp<DummiesCutOp>(); cut;
       cut = cut.getUpstream().getDefiningOp<DummiesCutOp>())
    onPath.push_back(cut);
  for (DummiesCutOp cut : llvm::reverse(onPath)) {
    OpBuilder builder(cut);
    auto axi4Cut = CutOp::create(builder, cut.getLoc(), port.getType(),
                                 cut.getClock(), cut.getReset(), port);
    for (NamedAttribute attr : getPulpConfig(cut))
      axi4Cut->setAttr(attr.getName(), attr.getValue());
    port = axi4Cut;
  }

  if (PortType needed = adapted.lookup(connection)) {
    Operation *consumer = connection->getOwner();
    auto [clock, reset] = domainOf(consumer);
    OpBuilder builder(consumer);

    // Re-widthing beats and re-tagging them are independent, each preserving
    // what the other changes. Widths come first, so the port between the two
    // carries the consumer's beats with the producer's tags.
    auto driven = cast<PortType>(port.getType());
    if (driven.getDataWidth() != needed.getDataWidth())
      port = DWConverterOp::create(
          builder, consumer->getLoc(),
          PortType::get(needed.getContext(), needed.getAddrWidth(),
                        needed.getDataWidth(), driven.getWriteIdWidth(),
                        driven.getReadIdWidth(), needed.getUserWidth(),
                        needed.getWindows(), driven.getOutstandingWrites(),
                        driven.getOutstandingReads()),
          clock, reset, port);
    if (driven.getWriteIdWidth() != needed.getWriteIdWidth() ||
        driven.getReadIdWidth() != needed.getReadIdWidth())
      port = IWConverterOp::create(builder, consumer->getLoc(), needed, clock,
                                   reset, port);
  }
  lowered.insert({connection, port});
}

/// Replace the network with AXI4 ops, and its external endpoints with ports of
/// the module describing it.
void NetworkLowering::emit() {
  for (DummiesExtManagerOp manager : managers) {
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    auto [name, arg] =
        module.appendInput(names.newName(manager.getName().value_or("manager")),
                           types[connection]);
    if (DictionaryAttr markers = getAtopMarkers(manager); !markers.empty())
      setPortAttrs(module,
                   module.getPortIdForInputId(module.getNumInputPorts() - 1),
                   markers);
    drive(connection, arg);
  }

  // A crossbar or remapper is built on placeholders for its inputs, which are
  // only all lowered once every stage is built.
  DenseMap<OpOperand *, Value> placeholders;
  auto placeholdersFor = [&](OpBuilder &builder, Operation *stage) {
    SmallVector<Value> inputs;
    for (OpOperand *connection : incomingConnections(stage)) {
      Value placeholder =
          UnrealizedConversionCastOp::create(builder, stage->getLoc(),
                                             types[connection], ValueRange())
              .getResult(0);
      placeholders.insert({connection, placeholder});
      inputs.push_back(placeholder);
    }
    return inputs;
  };

  for (DummiesIDRemapOp remap : remaps) {
    OpOperand *connection = outgoingConnections(remap.getDownstream()).front();
    OpOperand *upstream = incomingConnections(remap).front();
    OpBuilder builder(remap);
    auto axi4Remap = IDRemapOp::create(
        builder, remap.getLoc(), types[connection], remap.getClock(),
        remap.getReset(), placeholdersFor(builder, remap).front(),
        trackedIds(remap, types[upstream]));
    for (NamedAttribute attr : getPulpConfig(remap))
      axi4Remap->setAttr(attr.getName(), attr.getValue());
    drive(connection, axi4Remap);
  }

  for (DummiesXbarOp xbar : xbars) {
    SmallVector<OpOperand *> downstream =
        outgoingConnections(xbar.getDownstream());
    SmallVector<Type> results;
    for (OpOperand *connection : downstream)
      results.push_back(types[connection]);

    OpBuilder builder(xbar);
    auto axi4Xbar =
        XbarOp::create(builder, xbar.getLoc(), results, xbar.getClock(),
                       xbar.getReset(), placeholdersFor(builder, xbar));
    for (NamedAttribute attr : getPulpConfig(xbar))
      axi4Xbar->setAttr(attr.getName(), attr.getValue());

    // PULP connects every pair by default, so only a partial matrix is set.
    // Config already on the crossbar wins.
    auto connectivityName =
        builder.getStringAttr(Twine(kPulpConfigPrefix) + "Connectivity");
    ArrayRef<llvm::BitVector> rows = connectivity[xbar];
    if (!axi4Xbar->hasAttr(connectivityName) &&
        !llvm::all_of(rows,
                      [](const llvm::BitVector &row) { return row.all(); }))
      axi4Xbar->setAttr(connectivityName,
                        getConnectivityAttr(builder.getContext(), rows));

    for (auto [connection, result] :
         llvm::zip(downstream, axi4Xbar.getDownstream()))
      drive(connection, result);
  }

  for (auto [connection, placeholder] : placeholders) {
    Operation *cast = placeholder.getDefiningOp();
    placeholder.replaceAllUsesWith(lowered[connection]);
    cast->erase();
  }

  for (DummiesExtSubordinateOp subordinate : subordinates) {
    OpOperand *connection = incomingConnections(subordinate).front();
    module.appendOutput(
        names.newName(subordinate.getName().value_or("subordinate")),
        lowered[connection]);
    if (DictionaryAttr markers = getAtopMarkers(subordinate); !markers.empty())
      setPortAttrs(module,
                   module.getPortIdForOutputId(module.getNumOutputPorts() - 1),
                   markers);
  }

  // Every op lets go of its operands first, so the network is erased in any
  // order.
  SmallVector<Operation *> network;
  llvm::append_range(network, accesses);
  llvm::append_range(network, subordinates);
  llvm::append_range(network, cuts);
  llvm::append_range(network, remaps);
  llvm::append_range(network, xbars);
  llvm::append_range(network, managers);
  for (Operation *op : network)
    op->dropAllReferences();
  for (Operation *op : network)
    op->erase();
}

LogicalResult NetworkLowering::lower(const DenseSet<StringAttr> &instantiated) {
  if (instantiated.contains(module.getModuleNameAttr()))
    return module.emitOpError(
        "cannot lower a dummies network in an instantiated module; its "
        "external endpoints must become ports of a top-level module");
  if (failed(checkOneModule()))
    return failure();

  for (DummiesXbarOp xbar : xbars)
    if (xbar.getDownstream().use_empty())
      return xbar.emitOpError("must reach at least one subordinate");
  for (DummiesCutOp cut : cuts) {
    if (cut.getDownstream().use_empty())
      return cut.emitOpError("must reach a subordinate");
    if (isOnCutCycle(cut))
      return cut.emitOpError("is part of a cycle in the dummies network");
  }
  for (DummiesIDRemapOp remap : remaps)
    if (remap.getDownstream().use_empty())
      return remap.emitOpError("must reach a subordinate");

  for (DummiesExtManagerOp manager : managers)
    if (manager->hasAttr(kPulpAtopFilterAttr))
      return manager.emitOpError()
             << "is marked '" << kPulpAtopFilterAttr
             << "', but only a subordinate can have atomics filtered out in "
                "front of it";

  for (DummiesExtManagerOp manager : managers)
    for (Operation *user : manager.getPort().getUsers())
      if (failed(checkDomain(manager, user)))
        return failure();
  for (DummiesExtSubordinateOp subordinate : subordinates)
    if (failed(checkDomain(subordinate,
                           subordinate.getUpstream().getDefiningOp())))
      return failure();

  for (DummiesAccessesOp access : accesses)
    declared[access.getManager().getDefiningOp()].push_back(access);

  if (failed(inferTypes()) || failed(inferConnectivity()))
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
  using LowerAXI4DummiesToAXIBase::LowerAXI4DummiesToAXIBase;
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
    NetworkLowering lowering(hwModule, userWidth);
    if (!lowering.collect())
      continue;
    if (failed(lowering.lower(instantiated)))
      return signalPassFailure();
  }
}
