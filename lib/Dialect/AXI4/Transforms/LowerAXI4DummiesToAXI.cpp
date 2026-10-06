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

/// Warn where a subordinate can hold requests with fewer IDs, or fewer per ID,
/// than reach it through the port it presents, `presented`, which merges `ids`
/// onto its own IDs where they are narrower (this can't be caught
/// post-lowering)
static void warnBottleneck(DummiesExtSubordinateOp subordinate,
                           std::pair<uint64_t, uint64_t> ids,
                           PortType presented) {
  auto warn = [&](StringRef what, uint64_t held, uint64_t reaching) {
    if (held < reaching)
      subordinate.emitWarning() << "can " << what << " than reach it (" << held
                                << " < " << reaching << ")";
  };
  warn("track fewer write IDs", subordinate.getOutstandingWriteIds(),
       std::min(ids.first, uint64_t{1} << presented.getWriteIdWidth()));
  warn("track fewer read IDs", subordinate.getOutstandingReadIds(),
       std::min(ids.second, uint64_t{1} << presented.getReadIdWidth()));
  warn("hold fewer writes per ID", subordinate.getConcurrentWritesPerId(),
       presented.getConcurrentWritesPerId());
  warn("hold fewer reads per ID", subordinate.getConcurrentReadsPerId(),
       presented.getConcurrentReadsPerId());
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

/// A request count as a budget attribute, which admits at least one request.
static uint32_t clampToBudget(uint64_t count) {
  return static_cast<uint32_t>(
      std::clamp<uint64_t>(count, 1, std::numeric_limits<uint32_t>::max()));
}

/// The requests per ID left once `perId` per ID on `fromWidth` ID bits are
/// merged onto `toWidth` ID bits.
static uint32_t mergedPerId(uint32_t perId, uint32_t fromWidth,
                            uint32_t toWidth) {
  if (toWidth >= fromWidth)
    return perId;
  return perId << (fromWidth - toWidth);
}

/// The reads per ID left once a data width converter between `fromWidth` and
/// `toWidth` bits, if one is needed, has converted them one per ID at a time.
static uint32_t convertedReads(uint32_t reads, uint32_t fromWidth,
                               uint32_t toWidth) {
  return fromWidth == toWidth ? reads : std::min(reads, 1u);
}

/// The port type a subordinate presents over `windows` to a connection
/// carrying `driven`, with ID widths wide enough to tag every request it can
/// hold.
static PortType getSubordinatePortType(DummiesExtSubordinateOp subordinate,
                                       uint32_t userWidth,
                                       WindowSetAttr windows, PortType driven) {
  uint32_t writeIdWidth =
      llvm::Log2_64_Ceil(subordinate.getOutstandingWriteIds());
  uint32_t readIdWidth =
      llvm::Log2_64_Ceil(subordinate.getOutstandingReadIds());
  return PortType::get(
      subordinate.getContext(), subordinate.getAddrWidth(),
      subordinate.getDataWidth(), writeIdWidth, readIdWidth, userWidth, windows,
      mergedPerId(driven.getConcurrentWritesPerId(), driven.getWriteIdWidth(),
                  writeIdWidth),
      mergedPerId(convertedReads(driven.getConcurrentReadsPerId(),
                                 driven.getDataWidth(),
                                 subordinate.getDataWidth()),
                  driven.getReadIdWidth(), readIdWidth));
}

/// The same port type with different ID widths.
static PortType getPortTypeWithIdWidths(PortType port, uint32_t writeIdWidth,
                                        uint32_t readIdWidth) {
  return PortType::get(
      port.getContext(), port.getAddrWidth(), port.getDataWidth(), writeIdWidth,
      readIdWidth, port.getUserWidth(), port.getWindows(),
      port.getConcurrentWritesPerId(), port.getConcurrentReadsPerId());
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
                       port.getConcurrentWritesPerId(),
                       convertedReads(port.getConcurrentReadsPerId(),
                                      port.getDataWidth(), dataWidth));
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

/// The manager, crossbar or remapper driving a connection, past any cuts on
/// it.
static Operation *producerOf(OpOperand *connection) {
  Value port = connection->get();
  while (auto cut = port.getDefiningOp<DummiesCutOp>())
    port = cut.getUpstream();
  return port.getDefiningOp();
}

/// The data width a connection is driven with. A remapper passes on the one
/// reaching it.
static uint32_t dataWidthOf(OpOperand *connection) {
  return TypeSwitch<Operation *, uint32_t>(producerOf(connection))
      .Case<DummiesExtManagerOp, DummiesXbarOp>(
          [](auto producer) { return producer.getDataWidth(); })
      .Case<DummiesIDRemapOp>([](DummiesIDRemapOp remap) {
        return dataWidthOf(incomingConnections(remap).front());
      });
}

/// Whether a cut or remapper is fed only through other cuts and remappers,
/// around a cycle.
static bool isOnAdaptorCycle(Operation *op) {
  SmallPtrSet<Operation *, 8> seen;
  while (isa_and_nonnull<DummiesCutOp, DummiesIDRemapOp>(op)) {
    if (!seen.insert(op).second)
      return true;
    Value upstream = isa<DummiesCutOp>(op)
                         ? cast<DummiesCutOp>(op).getUpstream()
                         : cast<DummiesIDRemapOp>(op).getUpstream();
    op = upstream.getDefiningOp();
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Network lowering
//===----------------------------------------------------------------------===//

namespace {
/// Lowers the dummies network a module describes. Each connection - a use of a
/// dummies port, followed through any cuts - carries one `!axi4.port` type.
/// Windows are calculated by propagating them up from subordinates, and widths
/// are calculated by propagating them down from managers. The network may
/// loop, so long as every loop passes through an ID remapper.

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
  void findRoutes(OpOperand *connection, Operation *subordinate,
                  SmallPtrSetImpl<Operation *> &passed,
                  SmallVectorImpl<OpOperand *> &route,
                  SmallVectorImpl<SmallVector<OpOperand *>> &routes);
  FailureOr<SmallVector<WindowAttr>>
  windowsBelow(OpOperand *connection, uint32_t dataWidth,
               SmallPtrSetImpl<Operation *> &passed,
               const SmallPtrSetImpl<Operation *> &targeted);
  WindowSetAttr windowsOfRemapInput(OpOperand *connection);
  LogicalResult inferConnectionWindows();
  LogicalResult inferIdWidths();
  void clampUniqueIds();
  void inferCounts();
  PortType getRemapType(DummiesIDRemapOp remap);
  LogicalResult adaptToSubordinate(OpOperand *connection, PortType driven,
                                   const Twine &source);
  LogicalResult adaptToXbar(DummiesXbarOp xbar);
  LogicalResult inferTypes();
  LogicalResult routeAccesses();
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
  /// The subordinates the accesses routed through each connection target.
  DenseMap<OpOperand *, SmallPtrSet<Operation *, 4>> targeted;
  /// The windows each connection carries.
  DenseMap<OpOperand *, WindowSetAttr> windows;
  /// The IDs each remapper tracks.
  DenseMap<Operation *, uint32_t> uniqueIds;
  /// The write and read ID widths each manager, crossbar and remapper drives.
  DenseMap<Operation *, std::pair<uint32_t, uint32_t>> idWidths;
  /// The writes and reads per ID each connection carries.
  DenseMap<OpOperand *, std::pair<uint32_t, uint32_t>> perIds;
  /// The write and read IDs each connection carries.
  DenseMap<OpOperand *, std::pair<uint64_t, uint64_t>> ids;
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

/// Add the routes from `connection` to `subordinate` to `routes`, each the
/// connections along it, through no crossbar in `passed`. Stops at two, which
/// is already one too many.
void NetworkLowering::findRoutes(
    OpOperand *connection, Operation *subordinate,
    SmallPtrSetImpl<Operation *> &passed, SmallVectorImpl<OpOperand *> &route,
    SmallVectorImpl<SmallVector<OpOperand *>> &routes) {
  Operation *consumer = connection->getOwner();
  if (routes.size() == 2 ||
      (isa<DummiesXbarOp>(consumer) && passed.contains(consumer)))
    return;

  route.push_back(connection);
  if (consumer == subordinate) {
    routes.emplace_back(route.begin(), route.end());
  } else if (!isa<DummiesExtSubordinateOp>(consumer)) {
    passed.insert(consumer);
    for (OpOperand *below : outgoingConnections(downstreamOf(consumer)))
      findRoutes(below, subordinate, passed, route, routes);
    passed.erase(consumer);
  }
  route.pop_back();
}

/// The windows of the subordinates in `targeted` a connection reaches, or of
/// all of them if it is empty, in beats of `dataWidth`, along paths through no
/// crossbar in `passed`.
FailureOr<SmallVector<WindowAttr>>
NetworkLowering::windowsBelow(OpOperand *connection, uint32_t dataWidth,
                              SmallPtrSetImpl<Operation *> &passed,
                              const SmallPtrSetImpl<Operation *> &targeted) {
  Operation *consumer = connection->getOwner();

  // A remapper leaves the beats alone.
  if (auto remap = dyn_cast<DummiesIDRemapOp>(consumer))
    return windowsBelow(outgoingConnections(remap.getDownstream()).front(),
                        dataWidth, passed, targeted);

  // A crossbar carries the windows of everything below it, each of them
  // already in beats of its own data width.
  SmallVector<WindowAttr> presented;
  if (auto subordinate = dyn_cast<DummiesExtSubordinateOp>(consumer)) {
    if (!targeted.empty() && !targeted.contains(subordinate))
      return presented;
    llvm::append_range(presented, subordinate.getWindows().getWindows());
  } else {
    auto xbar = cast<DummiesXbarOp>(consumer);
    if (!passed.insert(xbar).second)
      return presented;
    for (OpOperand *below : outgoingConnections(xbar.getDownstream())) {
      FailureOr<SmallVector<WindowAttr>> windows =
          windowsBelow(below, xbar.getDataWidth(), passed, targeted);
      if (failed(windows))
        return failure();
      llvm::append_range(presented, *windows);
    }
    passed.erase(xbar);
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
  return windows;
}

/// The windows the connection into a remapper carries, which it carries on.
WindowSetAttr NetworkLowering::windowsOfRemapInput(OpOperand *connection) {
  if (WindowSetAttr known = windows.lookup(connection))
    return known;
  auto producer = cast<DummiesIDRemapOp>(producerOf(connection));
  WindowSetAttr carried =
      windowsOfRemapInput(incomingConnections(producer).front());
  windows.insert({connection, carried});
  return carried;
}

/// Give every connection its windows. A manager's are those its accesses grant
/// it. A crossbar's downstream port's are those of the subordinates the
/// accesses routed through it target, or of every subordinate below it if none
/// is routed through it, reached without passing back through the crossbar or
/// any other twice.
LogicalResult NetworkLowering::inferConnectionWindows() {
  for (DummiesExtManagerOp manager : managers) {
    FailureOr<WindowSetAttr> granted = inferWindows(manager, declared[manager]);
    if (failed(granted))
      return failure();
    windows.insert({outgoingConnections(manager.getPort()).front(), *granted});
  }

  for (DummiesXbarOp xbar : xbars) {
    for (auto [index, connection] :
         llvm::enumerate(outgoingConnections(xbar.getDownstream()))) {
      SmallPtrSet<Operation *, 8> passed = {xbar};
      FailureOr<SmallVector<WindowAttr>> below = windowsBelow(
          connection, xbar.getDataWidth(), passed, targeted[connection]);
      if (failed(below))
        return failure();
      if (below->empty())
        return xbar.emitOpError()
               << "downstream port #" << index
               << " reaches no subordinate without looping back through a "
                  "crossbar";

      // The address widths below are only checked once their crossbars are
      // lowered, so the windows they present can be too wide for this one.
      auto emitError = [&]() { return xbar.emitOpError(); };
      if (failed(verifyWindowsFit(emitError, "", xbar.getAddrWidth(), *below)))
        return failure();
      windows.insert(
          {connection, WindowSetAttr::get(module.getContext(), *below)});
    }
  }

  for (DummiesIDRemapOp remap : remaps)
    windows.insert({outgoingConnections(remap.getDownstream()).front(),
                    windowsOfRemapInput(incomingConnections(remap).front())});
  return success();
}

/// Give every manager, crossbar and remapper the ID widths it drives. A
/// remapper's are its own, so the crossbars are ordered as if its upstream
/// connection were cut, and any loop left over has no remapper on it.
LogicalResult NetworkLowering::inferIdWidths() {
  idWidths.clear();
  // An endpoint needs enough ID bits to tell apart every ID it can have
  // outstanding.
  for (DummiesExtManagerOp manager : managers)
    idWidths[manager] = {llvm::Log2_64_Ceil(manager.getOutstandingWriteIds()),
                         llvm::Log2_64_Ceil(manager.getOutstandingReadIds())};
  for (DummiesIDRemapOp remap : remaps) {
    uint32_t idWidth = llvm::Log2_64_Ceil(uniqueIds[remap]);
    idWidths[remap] = {idWidth, idWidth};
  }

  SmallVector<DummiesXbarOp> pending(xbars);
  while (!pending.empty()) {
    SmallVector<DummiesXbarOp> waiting;
    for (DummiesXbarOp xbar : pending) {
      SmallVector<OpOperand *> upstream = incomingConnections(xbar);
      if (!llvm::all_of(upstream, [&](OpOperand *connection) {
            return idWidths.contains(producerOf(connection));
          })) {
        waiting.push_back(xbar);
        continue;
      }

      // Transactions are tagged with the index of the manager they came from,
      // so the downstream ports carry wider IDs than the widest upstream one.
      uint32_t writeIdWidth = 0, readIdWidth = 0;
      for (OpOperand *connection : upstream) {
        auto [writes, reads] = idWidths[producerOf(connection)];
        writeIdWidth = std::max(writeIdWidth, writes);
        readIdWidth = std::max(readIdWidth, reads);
      }
      uint32_t tagBits = llvm::Log2_64_Ceil(upstream.size());
      idWidths[xbar] = {writeIdWidth + tagBits, readIdWidth + tagBits};
    }
    if (waiting.size() == pending.size())
      return waiting.front().emitOpError(
          "is part of a loop with no ID remapper, around which its IDs would "
          "grow without bound");
    pending = waiting;
  }
  return success();
}

/// Have each remapper track no more IDs than its upstream port can carry.
/// Tracking fewer narrows the IDs below it, and so possibly the ports reaching
/// other remappers, so it repeats until no remapper changes. Each only ever
/// tracks fewer, so it ends.
void NetworkLowering::clampUniqueIds() {
  bool clamped = true;
  while (clamped) {
    clamped = false;
    for (DummiesIDRemapOp remap : remaps) {
      auto [writes, reads] =
          idWidths[producerOf(incomingConnections(remap).front())];
      uint64_t carried = uint64_t{1} << std::min(writes, reads);
      if (uniqueIds[remap] > carried) {
        uniqueIds[remap] = static_cast<uint32_t>(carried);
        clamped = true;
      }
    }
    if (clamped)
      (void)inferIdWidths();
  }
}

/// Give every connection the IDs it carries, and the writes and reads per ID.
/// A crossbar's downstream port carries what the managers that can address it
/// issue there, each keeping its IDs apart, and no more per ID than its budget,
/// and a remapper's no more IDs than it tracks or per ID than its budget. A
/// data width converter in front of a crossbar converts one read per ID at a
/// time. Around a loop the counts depend
/// on each other, so starting from none it repeats until none changes. Counts
/// only grow, per ID to no more than a manager issues and in IDs to no more
/// than the remapper on every loop tracks, so it ends.
void NetworkLowering::inferCounts() {
  for (DummiesExtManagerOp manager : managers) {
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    perIds[connection] = {manager.getConcurrentWritesPerId(),
                          manager.getConcurrentReadsPerId()};
    ids[connection] = {manager.getOutstandingWriteIds(),
                       manager.getOutstandingReadIds()};
  }

  bool changed = true;
  while (changed) {
    changed = false;
    auto update = [&](OpOperand *connection,
                      std::pair<uint32_t, uint32_t> perId,
                      std::pair<uint64_t, uint64_t> carried) {
      auto [perIdIt, newPerId] = perIds.try_emplace(connection, perId);
      auto [idsIt, newIds] = ids.try_emplace(connection, carried);
      if (!newPerId && !newIds && perIdIt->second == perId &&
          idsIt->second == carried)
        return;
      perIdIt->second = perId;
      idsIt->second = carried;
      changed = true;
    };

    for (DummiesXbarOp xbar : xbars) {
      for (OpOperand *connection : outgoingConnections(xbar.getDownstream())) {
        uint32_t writesPerId = 0, readsPerId = 0;
        uint64_t writeIds = 0, readIds = 0;
        for (OpOperand *above : incomingConnections(xbar)) {
          if (!windows[above].overlaps(windows[connection]))
            continue;
          auto [aboveWritesPerId, aboveReadsPerId] = perIds.lookup(above);
          auto [aboveWriteIds, aboveReadIds] = ids.lookup(above);
          writesPerId = std::max(writesPerId, aboveWritesPerId);
          readsPerId = std::max(
              readsPerId, convertedReads(aboveReadsPerId, dataWidthOf(above),
                                         xbar.getDataWidth()));
          writeIds += aboveWriteIds;
          readIds += aboveReadIds;
        }
        uint32_t budget = xbar.getUpstreamConcurrentPerId();
        update(connection,
               {std::min(writesPerId, budget), std::min(readsPerId, budget)},
               {writeIds, readIds});
      }
    }

    for (DummiesIDRemapOp remap : remaps) {
      OpOperand *upstream = incomingConnections(remap).front();
      auto [writeIds, readIds] = ids.lookup(upstream);
      auto [writesPerId, readsPerId] = perIds.lookup(upstream);
      uint64_t tracked = uniqueIds[remap];
      uint32_t budget = remap.getConcurrentPerId();
      update(outgoingConnections(remap.getDownstream()).front(),
             {std::min(writesPerId, budget), std::min(readsPerId, budget)},
             {std::min(writeIds, tracked), std::min(readIds, tracked)});
    }
  }
}

/// The port type a remapper drives: the one reaching it, with the IDs and
/// requests of its own.
PortType NetworkLowering::getRemapType(DummiesIDRemapOp remap) {
  OpOperand *connection = outgoingConnections(remap.getDownstream()).front();
  if (PortType known = types.lookup(connection))
    return known;

  OpOperand *upstream = incomingConnections(remap).front();
  PortType reaching = types.lookup(upstream);
  if (!reaching)
    reaching = getRemapType(cast<DummiesIDRemapOp>(producerOf(upstream)));
  auto [writeIdWidth, readIdWidth] = idWidths[remap];
  auto [writes, reads] = perIds[connection];
  PortType driven = PortType::get(
      module.getContext(), reaching.getAddrWidth(), reaching.getDataWidth(),
      writeIdWidth, readIdWidth, userWidth, windows[connection], writes, reads);
  types.insert({connection, driven});
  return driven;
}

/// Adapt a connection carrying `driven` straight to the subordinate consuming
/// it, if one does. The subordinate presents a port of its own: its own data
/// width, and its own ID widths, log2 of the IDs it can hold.
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
  PortType port =
      getSubordinatePortType(subordinate, userWidth, *served, driven);
  warnBottleneck(subordinate, ids.lookup(connection), port);
  if (needsConverter(port, driven.getDataWidth(), driven.getWriteIdWidth(),
                     driven.getReadIdWidth()))
    adapted.insert({connection, port});
  return success();
}

/// Adapt each of a crossbar's upstream connections to the widths it routes
/// over.
LogicalResult NetworkLowering::adaptToXbar(DummiesXbarOp xbar) {
  // A crossbar routes, it does not re-address, so it must agree with
  // everything it connects on the address width.
  SmallVector<OpOperand *> upstream = incomingConnections(xbar);
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
  return success();
}

/// Give every connection in the network a port type, and the converters its
/// consumer needs.
LogicalResult NetworkLowering::inferTypes() {
  for (DummiesIDRemapOp remap : remaps)
    uniqueIds[remap] = remap.getMaxUniqueIds();
  if (failed(inferIdWidths()) || failed(routeAccesses()) ||
      failed(inferConnectionWindows()))
    return failure();
  clampUniqueIds();
  inferCounts();

  for (DummiesExtManagerOp manager : managers) {
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    auto [writeIdWidth, readIdWidth] = idWidths[manager];
    types.insert(
        {connection,
         PortType::get(module.getContext(), manager.getAddrWidth(),
                       manager.getDataWidth(), writeIdWidth, readIdWidth,
                       userWidth, windows[connection], perIds[connection].first,
                       perIds[connection].second)});
  }
  for (DummiesXbarOp xbar : xbars) {
    auto [writeIdWidth, readIdWidth] = idWidths[xbar];
    for (OpOperand *connection : outgoingConnections(xbar.getDownstream())) {
      auto [writes, reads] = perIds[connection];
      types.insert(
          {connection,
           PortType::get(module.getContext(), xbar.getAddrWidth(),
                         xbar.getDataWidth(), writeIdWidth, readIdWidth,
                         userWidth, windows[connection], writes, reads)});
    }
  }
  for (DummiesIDRemapOp remap : remaps)
    getRemapType(remap);

  for (DummiesExtManagerOp manager : managers) {
    OpOperand *connection = outgoingConnections(manager.getPort()).front();
    if (failed(adaptToSubordinate(connection, types[connection], "manager")))
      return failure();
  }
  for (DummiesXbarOp xbar : xbars) {
    if (failed(adaptToXbar(xbar)))
      return failure();
    for (OpOperand *connection : outgoingConnections(xbar.getDownstream()))
      if (failed(adaptToSubordinate(connection, types[connection], "crossbar")))
        return failure();
  }
  for (DummiesIDRemapOp remap : remaps) {
    OpOperand *connection = outgoingConnections(remap.getDownstream()).front();
    if (failed(adaptToSubordinate(connection, types[connection], "remapper")))
      return failure();
  }
  return success();
}

/// Route each access to the subordinate it targets, along the one route from
/// its manager through no crossbar twice. Each connection on the route targets
/// the subordinate, and each crossbar on it connects the upstream port the
/// route enters by to the downstream port it leaves by.
LogicalResult NetworkLowering::routeAccesses() {
  for (DummiesXbarOp xbar : xbars)
    connectivity[xbar].assign(
        xbar.getUpstream().size(),
        llvm::BitVector(outgoingConnections(xbar.getDownstream()).size()));

  for (DummiesAccessesOp access : accesses) {
    auto manager = access.getManager().getDefiningOp<DummiesExtManagerOp>();
    Operation *subordinate = access.getSubordinate().getDefiningOp();
    SmallVector<OpOperand *> connections =
        outgoingConnections(manager.getPort());
    SmallPtrSet<Operation *, 8> passed;
    SmallVector<OpOperand *> route;
    SmallVector<SmallVector<OpOperand *>> routes;
    if (!connections.empty())
      findRoutes(connections.front(), subordinate, passed, route, routes);
    if (routes.empty())
      return access.emitOpError(
          "declares an access to a subordinate the manager cannot reach");
    if (routes.size() > 1)
      return access.emitOpError("is ambiguous: the manager reaches the "
                                "subordinate by more than one route");

    for (auto [above, below] :
         llvm::zip(routes.front(), llvm::drop_begin(routes.front()))) {
      targeted[below].insert(subordinate);
      auto xbar = dyn_cast<DummiesXbarOp>(above->getOwner());
      if (!xbar)
        continue;
      unsigned upstream =
          above->getOperandNumber() - xbar.getUpstream().getBeginOperandIndex();
      SmallVector<OpOperand *> downstream =
          outgoingConnections(xbar.getDownstream());
      connectivity[xbar][upstream].set(llvm::find(downstream, below) -
                                       downstream.begin());
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

    // The converters are sized for the IDs the connection carries. A data
    // width converter feeding a subordinate converts no more read IDs than the
    // subordinate holds.
    auto driven = cast<PortType>(port.getType());
    auto [writeIds, readIds] = ids.lookup(connection);
    uint64_t convertedReadIds = readIds;
    if (auto subordinate = dyn_cast<DummiesExtSubordinateOp>(consumer))
      convertedReadIds =
          std::min<uint64_t>(readIds, subordinate.getOutstandingReadIds());

    // Re-widthing beats and re-tagging them are independent, each preserving
    // what the other changes. Widths come first, so the port between the two
    // carries the consumer's beats with the producer's tags.
    if (driven.getDataWidth() != needed.getDataWidth())
      port = DWConverterOp::create(
          builder, consumer->getLoc(),
          PortType::get(needed.getContext(), needed.getAddrWidth(),
                        needed.getDataWidth(), driven.getWriteIdWidth(),
                        driven.getReadIdWidth(), needed.getUserWidth(),
                        needed.getWindows(), driven.getConcurrentWritesPerId(),
                        convertedReads(driven.getConcurrentReadsPerId(),
                                       driven.getDataWidth(),
                                       needed.getDataWidth())),
          clock, reset, port, clampToBudget(convertedReadIds));
    if (driven.getWriteIdWidth() != needed.getWriteIdWidth() ||
        driven.getReadIdWidth() != needed.getReadIdWidth()) {
      // It can track no more IDs than the narrower of the two upstream widths
      // gives.
      uint32_t idWidth =
          std::min(driven.getWriteIdWidth(), driven.getReadIdWidth());
      uint64_t uniqueIds =
          std::min(std::max(writeIds, readIds), uint64_t{1} << idWidth);
      uint32_t perId = std::max(driven.getConcurrentWritesPerId(),
                                driven.getConcurrentReadsPerId());
      port = IWConverterOp::create(builder, consumer->getLoc(), needed, clock,
                                   reset, port, clampToBudget(uniqueIds),
                                   std::max(perId, 1u));
    }
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
    OpBuilder builder(remap);
    auto axi4Remap = IDRemapOp::create(
        builder, remap.getLoc(), types[connection], remap.getClock(),
        remap.getReset(), placeholdersFor(builder, remap).front(),
        uniqueIds[remap], remap.getConcurrentPerId());
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
    auto axi4Xbar = XbarOp::create(
        builder, xbar.getLoc(), results, xbar.getClock(), xbar.getReset(),
        placeholdersFor(builder, xbar), xbar.getUpstreamConcurrentPerIdAttr());
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
  // A loop of nothing but cuts and remappers has no manager to drive it.
  for (DummiesIDRemapOp remap : remaps) {
    if (remap.getDownstream().use_empty())
      return remap.emitOpError("must reach a subordinate");
    if (isOnAdaptorCycle(remap))
      return remap.emitOpError("is part of a cycle in the dummies network");
  }
  for (DummiesCutOp cut : cuts) {
    if (cut.getDownstream().use_empty())
      return cut.emitOpError("must reach a subordinate");
    if (isOnAdaptorCycle(cut))
      return cut.emitOpError("is part of a cycle in the dummies network");
  }

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

  if (failed(inferTypes()))
    return failure();

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
