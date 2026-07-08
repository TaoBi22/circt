//===- AXI4ToHW.cpp - Translate AXI4 into HW
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main AXI4 to HW Conversion Pass Implementation. It lowers the
// abstract AXI4 network dialect into instantiated hardware, wiring each node
// and crossbar together according to the network's SSA edges.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AXI4ToHW.h"
#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_AXI4TOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::axi4;
using namespace circt::hw;

//===----------------------------------------------------------------------===//
// Canonical channel representation
//===----------------------------------------------------------------------===//

namespace {

/// One network signal that may or may not be a backedge.
struct Wire {
  Value value;
  std::optional<Backedge> backedge;
  bool isBackedge() const { return backedge.has_value(); }
};

/// One AXI4 channel's three signals. `ready` flows opposite to
/// `payload`/`valid`.
struct ChannelWires {
  Wire payload;
  Wire valid;
  Wire ready;
};

/// A port's five channels, in canonical order (AW, W, B, AR, R).
using PortWires = SmallVector<ChannelWires, 5>;

/// The two sides of one AXI edge, keyed by the consuming operand. Each side is
/// filled when its endpoint is lowered; both exist by the time we connect.
struct EdgeWires {
  PortWires producer;
  PortWires consumer;
};

// Fixed AXI4 field widths (bits).
constexpr unsigned kLenWidth = 8;
constexpr unsigned kSizeWidth = 3;
constexpr unsigned kBurstWidth = 2;
constexpr unsigned kLockWidth = 1;
constexpr unsigned kCacheWidth = 4;
constexpr unsigned kProtWidth = 3;
constexpr unsigned kQosWidth = 4;
constexpr unsigned kRegionWidth = 4;
constexpr unsigned kRespWidth = 2;
constexpr unsigned kLastWidth = 1;

enum class AXI4Channel { AW, W, B, AR, R };
constexpr unsigned kNumChannels = 5;

/// Per-channel lowering metadata. `token` is the port-name infix ("aw", "w",
/// ...); `isRequest` is true for channels a manager drives (AW, W, AR).
struct ChannelInfo {
  AXI4Channel channel;
  StringRef token;
  bool isRequest;
};
const ChannelInfo kChannelInfos[kNumChannels] = {{AXI4Channel::AW, "aw", true},
                                                 {AXI4Channel::W, "w", true},
                                                 {AXI4Channel::B, "b", false},
                                                 {AXI4Channel::AR, "ar", true},
                                                 {AXI4Channel::R, "r", false}};

/// Build the `hw.struct` payload type for one channel of an `!axi4.port`.
hw::StructType getChannelPayloadType(PortType port, AXI4Channel channel) {
  MLIRContext *ctx = port.getContext();
  auto intTy = [&](unsigned width) { return IntegerType::get(ctx, width); };
  auto field = [&](StringRef name, unsigned width) {
    return hw::StructType::FieldInfo{StringAttr::get(ctx, name), intTy(width)};
  };

  SmallVector<hw::StructType::FieldInfo> fields;
  switch (channel) {
  case AXI4Channel::AW:
  case AXI4Channel::AR:
    fields = {
        field("id", port.getIdWidth()), field("addr", port.getAddressWidth()),
        field("len", kLenWidth),        field("size", kSizeWidth),
        field("burst", kBurstWidth),    field("lock", kLockWidth),
        field("cache", kCacheWidth),    field("prot", kProtWidth),
        field("qos", kQosWidth),        field("region", kRegionWidth)};
    break;
  case AXI4Channel::W:
    fields = {field("data", port.getDataWidth()),
              field("strb", port.getDataWidth() / 8),
              field("last", kLastWidth)};
    break;
  case AXI4Channel::B:
    fields = {field("id", port.getIdWidth()), field("resp", kRespWidth)};
    break;
  case AXI4Channel::R:
    fields = {field("id", port.getIdWidth()),
              field("data", port.getDataWidth()), field("resp", kRespWidth),
              field("last", kLastWidth)};
    break;
  }
  return hw::StructType::get(ctx, fields);
}

//===----------------------------------------------------------------------===//
// Pre-pass rejections
//===----------------------------------------------------------------------===//

LogicalResult checkNetwork(ModuleOp module) {
  bool failed = false;

  Value commonClock;
  auto checkClock = [&](Operation *op, Value clock) {
    if (!commonClock)
      commonClock = clock;
    else if (clock != commonClock) {
      op->emitError("multiple clock domains are not yet supported");
      failed = true;
    }
  };

  auto checkUsers = [&](Operation *op) {
    for (Operation *user : op->getUsers())
      if (!isa_and_nonnull<AXI4Dialect>(user->getDialect())) {
        op->emitError("results of axi4 operations may only be used by axi4 "
                      "operations");
        failed = true;
        break;
      }
  };

  auto checkMapping = [&](Operation *op,
                          std::optional<AXI4PortMappingAttrInterface> mapping) {
    if (mapping && !isa<PortWiresAttr>(*mapping)) {
      op->emitError(
          "only the 'port_wires' port_mapping is currently supported");
      failed = true;
    }
  };

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ManagerPortOp>([&](ManagerPortOp mgr) {
          checkClock(mgr, mgr.getClock());
          checkUsers(mgr);
          if (!mgr.getNode()) {
            mgr.emitError("nodeless ports are not yet supported");
            failed = true;
            return;
          }
          if (mgr.getPort().use_empty()) {
            mgr.emitError("axi4.manager_ports with no uses are not yet "
                          "supported");
            failed = true;
          }
          checkMapping(mgr, mgr.getPortMapping());
        })
        .Case<SubordinatePortOp>([&](SubordinatePortOp sub) {
          checkClock(sub, sub.getClock());
          if (!sub.getNode()) {
            sub.emitError("nodeless ports are not yet supported");
            failed = true;
            return;
          }
          checkMapping(sub, sub.getPortMapping());
        })
        .Case<XbarOp>([&](XbarOp xbar) {
          checkClock(xbar, xbar.getClock());
          checkUsers(xbar);
        })
        .Case<NodeOp>([&](NodeOp node) { checkUsers(node); });
  });

  return failure(failed);
}

//===----------------------------------------------------------------------===//
// Network synthesis
//===----------------------------------------------------------------------===//

/// Bind the backedge side of an edge to the value driven by the other side.
void connectWire(Wire &a, Wire &b) {
  assert(a.isBackedge() != b.isBackedge() &&
         "exactly one side of an AXI4 wire must be a backedge");
  if (a.isBackedge())
    a.backedge->setValue(b.value);
  else
    b.backedge->setValue(a.value);
}

void connectPorts(PortWires &a, PortWires &b) {
  for (unsigned i = 0; i < kNumChannels; ++i) {
    connectWire(a[i].payload, b[i].payload);
    connectWire(a[i].valid, b[i].valid);
    connectWire(a[i].ready, b[i].ready);
  }
}

/// Interface for wiring one port group's channels onto a module's physical
/// ports. `FromModule` binds a module output, `ToModule` drives a module input
class MappingLowerer {
public:
  virtual ~MappingLowerer() = default;
  virtual LogicalResult registerPayloadFromModule(const ChannelInfo &ci,
                                                  hw::StructType payloadTy,
                                                  Wire &payload) = 0;
  virtual LogicalResult registerPayloadToModule(const ChannelInfo &ci,
                                                hw::StructType payloadTy,
                                                Wire &payload) = 0;
  virtual LogicalResult registerValidFromModule(const ChannelInfo &ci,
                                                Wire &valid) = 0;
  virtual LogicalResult registerValidToModule(const ChannelInfo &ci,
                                              Wire &valid) = 0;
  virtual LogicalResult registerReadyFromModule(const ChannelInfo &ci,
                                                Wire &ready) = 0;
  virtual LogicalResult registerReadyToModule(const ChannelInfo &ci,
                                              Wire &ready) = 0;
};

/// Lowers a port group using the flat-scalar `port_wires` mapping.
class PortWiresMappingLowerer : public MappingLowerer {
public:
  PortWiresMappingLowerer(
      Operation *portOp, bool isManager, PortWiresAttr mapping,
      DenseMap<StringRef, PortInfo> &portByName,
      DenseMap<StringRef, Value> &inputByName,
      SmallVectorImpl<std::pair<Wire *, unsigned>> &outs,
      SmallVectorImpl<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
          &outStructs,
      ImplicitLocOpBuilder &builder)
      : portOp(portOp),
        base(((isManager ? "m_axi_" : "s_axi_") + mapping.getName() + "_")
                 .str()),
        portByName(portByName), inputByName(inputByName), outs(outs),
        outStructs(outStructs), builder(builder) {}

  LogicalResult registerPayloadFromModule(const ChannelInfo &ci,
                                          hw::StructType payloadTy,
                                          Wire &payload) override {
    std::string cbase = getChannelBase(ci);
    SmallVector<unsigned> fieldArgs;
    for (auto &field : payloadTy.getElements()) {
      auto port = lookup(cbase + field.name.getValue(), ModulePort::Output);
      if (!port)
        return failure();
      fieldArgs.push_back(port->argNum);
    }
    outStructs.emplace_back(&payload, payloadTy, std::move(fieldArgs));
    return success();
  }

  LogicalResult registerPayloadToModule(const ChannelInfo &ci,
                                        hw::StructType payloadTy,
                                        Wire &payload) override {
    std::string cbase = getChannelBase(ci);
    for (auto &field : payloadTy.getElements()) {
      auto port = lookup(cbase + field.name.getValue(), ModulePort::Input);
      if (!port)
        return failure();
      inputByName[port->name.getValue()] =
          hw::StructExtractOp::create(builder, payload.value, field.name);
    }
    return success();
  }

  LogicalResult registerValidFromModule(const ChannelInfo &ci,
                                        Wire &valid) override {
    return registerScalarFromModule(getChannelBase(ci) + "valid", valid);
  }

  LogicalResult registerValidToModule(const ChannelInfo &ci,
                                      Wire &valid) override {
    return registerScalarToModule(getChannelBase(ci) + "valid", valid);
  }

  LogicalResult registerReadyFromModule(const ChannelInfo &ci,
                                        Wire &ready) override {
    return registerScalarFromModule(getChannelBase(ci) + "ready", ready);
  }

  LogicalResult registerReadyToModule(const ChannelInfo &ci,
                                      Wire &ready) override {
    return registerScalarToModule(getChannelBase(ci) + "ready", ready);
  }

private:
  std::optional<PortInfo> lookup(const Twine &nameT,
                                 ModulePort::Direction dir) {
    std::string name = nameT.str();
    auto it = portByName.find(name);
    if (it == portByName.end()) {
      portOp->emitError("referenced module has no port '")
          << name << "' required by the port_wires mapping";
      return std::nullopt;
    }
    if (it->second.dir != dir) {
      portOp->emitError("module port '") << name << "' has the wrong direction";
      return std::nullopt;
    }
    return it->second;
  }

  std::string getChannelBase(const ChannelInfo &ci) const {
    return base + ci.token.str();
  }

  LogicalResult registerScalarFromModule(const Twine &name, Wire &wire) {
    auto port = lookup(name, ModulePort::Output);
    if (!port)
      return failure();
    outs.emplace_back(&wire, port->argNum);
    return success();
  }

  LogicalResult registerScalarToModule(const Twine &name, Wire &wire) {
    auto port = lookup(name, ModulePort::Input);
    if (!port)
      return failure();
    inputByName[port->name.getValue()] = wire.value;
    return success();
  }

  /// Op used for diagnostics.
  Operation *portOp;
  /// `m_axi_`/`s_axi_` port name prefix.
  std::string base;
  /// Module ports by name.
  DenseMap<StringRef, PortInfo> &portByName;
  /// Instance input drivers.
  DenseMap<StringRef, Value> &inputByName;
  /// Scalar outputs resolved after instantiation.
  SmallVectorImpl<std::pair<Wire *, unsigned>> &outs;
  /// Payload outputs assembled after instantiation.
  SmallVectorImpl<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
      &outStructs;
  /// Insertion point for helper ops.
  ImplicitLocOpBuilder &builder;
};

/// Populate `wires` for one port group, walking the five channels and placing
/// backedges by direction; `mapping` supplies the mapping-specific port wiring.
LogicalResult populatePortGroup(bool isManager, PortType portType,
                                ImplicitLocOpBuilder &builder,
                                BackedgeBuilder &bb, PortWires &wires,
                                MappingLowerer &mapping) {
  Type i1 = builder.getI1Type();

  wires.resize(kNumChannels);
  for (auto [idx, ci] : llvm::enumerate(kChannelInfos)) {
    hw::StructType payloadTy = getChannelPayloadType(portType, ci.channel);
    bool payloadDrivenByModule = (isManager == ci.isRequest);
    ChannelWires &cw = wires[idx];

    if (payloadDrivenByModule) {
      if (failed(mapping.registerPayloadFromModule(ci, payloadTy, cw.payload)))
        return failure();
      if (failed(mapping.registerValidFromModule(ci, cw.valid)))
        return failure();

      Backedge ready = bb.get(i1);
      cw.ready = {ready, ready};
      if (failed(mapping.registerReadyToModule(ci, cw.ready)))
        return failure();
      continue;
    }

    Backedge payload = bb.get(payloadTy);
    cw.payload = {payload, payload};
    if (failed(mapping.registerPayloadToModule(ci, payloadTy, cw.payload)))
      return failure();

    Backedge valid = bb.get(i1);
    cw.valid = {valid, valid};
    if (failed(mapping.registerValidToModule(ci, cw.valid)))
      return failure();

    if (failed(mapping.registerReadyFromModule(ci, cw.ready)))
      return failure();
  }
  return success();
}

class NetworkLowering {
public:
  NetworkLowering(ModuleOp module)
      : module(module), builder(module.getLoc(), module.getContext()),
        bb(builder, module.getLoc()) {
    builder.setInsertionPointToEnd(module.getBody());
  }

  LogicalResult run();

private:
  LogicalResult lowerNetwork();
  LogicalResult lowerNode(NodeOp node);
  /// Populate `wires` (and the instance's input drivers) for one port group.
  LogicalResult addPortGroup(
      Operation *portOp, bool isManager, PortType portType,
      AXI4PortMappingAttrInterface mapping,
      DenseMap<StringRef, PortInfo> &portByName,
      DenseMap<StringRef, Value> &inputByName, PortWires &wires,
      SmallVectorImpl<std::pair<Wire *, unsigned>> &outs,
      SmallVectorImpl<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
          &outStructs);
  /// Materialize an `!axi4.clock` value as the module clock port's type.
  Value materializeClock(Value axiClock, Type portType);

  ModuleOp module;
  ImplicitLocOpBuilder builder;
  BackedgeBuilder bb;
  DenseMap<OpOperand *, EdgeWires> edges;
  DenseMap<Value, Value> clockCache;
  unsigned instanceCounter = 0;
};

// Materialize clocks as i1 since that's what'll be coming through the Slang
// frontend (at time of writing)
Value NetworkLowering::materializeClock(Value axiClock, Type portType) {
  if (!portType.isInteger(1))
    return {};
  Value &clock = clockCache[axiClock];
  if (!clock)
    clock = UnrealizedConversionCastOp::create(builder, portType, axiClock)
                .getResult(0);
  return clock;
}

LogicalResult NetworkLowering::addPortGroup(
    Operation *portOp, bool isManager, PortType portType,
    AXI4PortMappingAttrInterface mapping,
    DenseMap<StringRef, PortInfo> &portByName,
    DenseMap<StringRef, Value> &inputByName, PortWires &wires,
    SmallVectorImpl<std::pair<Wire *, unsigned>> &outs,
    SmallVectorImpl<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
        &outStructs) {
  if (auto portWires = dyn_cast<PortWiresAttr>(mapping)) {
    PortWiresMappingLowerer lowerer(portOp, isManager, portWires, portByName,
                                    inputByName, outs, outStructs, builder);
    return populatePortGroup(isManager, portType, builder, bb, wires, lowerer);
  }

  return portOp->emitError(
      "only the 'port_wires' port_mapping is currently supported");
}

LogicalResult NetworkLowering::lowerNode(NodeOp node) {
  SmallVector<Operation *> ports(node.getNode().getUsers());
  if (ports.empty())
    return success();

  // The NodeOp verifier guarantees the symbol resolves to an hw.module.
  auto moduleOp = module.lookupSymbol<hw::HWModuleLike>(node.getModuleAttr());
  assert(moduleOp && "node symbol does not map to module");

  DenseMap<StringRef, PortInfo> portByName;
  for (auto &p : moduleOp.getPortList())
    portByName[p.getName()] = p;

  builder.setLoc(node.getLoc());

  // Backedge-side wires live in `allWires` (stable, reserved) so the deferred
  // output pointers below stay valid until we resolve them post-instantiation.
  SmallVector<PortWires, 0> allWires;
  allWires.reserve(ports.size());
  DenseMap<StringRef, Value> inputByName;
  SmallVector<std::pair<Wire *, unsigned>> outs;
  SmallVector<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
      outStructs;

  for (Operation *portOp : ports) {
    bool isManager = isa<ManagerPortOp>(portOp);
    Value axiClock;
    PortType portType;
    AXI4PortMappingAttrInterface mapping;
    if (auto mgr = dyn_cast<ManagerPortOp>(portOp)) {
      axiClock = mgr.getClock();
      portType = cast<PortType>(mgr.getPort().getType());
      mapping = *mgr.getPortMapping();
    } else {
      auto sub = cast<SubordinatePortOp>(portOp);
      axiClock = sub.getClock();
      portType = cast<PortType>(sub.getUpstream().getType());
      mapping = *sub.getPortMapping();
    }

    allWires.emplace_back();
    if (failed(addPortGroup(portOp, isManager, portType, mapping, portByName,
                            inputByName, allWires.back(), outs, outStructs)))
      return failure();

    auto clockPort = portByName.find(mapping.getClockPort());
    if (clockPort == portByName.end())
      return portOp->emitError("referenced module has no clock port '")
             << mapping.getClockPort() << "'";
    Value clock = materializeClock(axiClock, clockPort->second.type);
    if (!clock)
      return portOp->emitError("unsupported clock port type");
    inputByName[clockPort->second.name.getValue()] = clock;
  }

  // Assemble instance inputs in the module's input-port order.
  SmallVector<Value> inputs;
  for (auto &p : moduleOp.getPortList()) {
    if (!p.isInput())
      continue;
    auto it = inputByName.find(p.getName());
    if (it == inputByName.end())
      return node.emitError("module input port '")
             << p.getName() << "' is not driven by any AXI4 port";
    inputs.push_back(it->second);
  }

  auto inst = hw::InstanceOp::create(
      builder, moduleOp,
      builder.getStringAttr(node.getModule() + "_" + Twine(instanceCounter++)),
      inputs);

  // Resolve deferred module outputs into real driving values.
  for (auto &[wire, argNum] : outs)
    wire->value = inst.getResult(argNum);
  for (auto &[wire, structTy, fieldArgs] : outStructs) {
    SmallVector<Value> fields;
    for (unsigned argNum : fieldArgs)
      fields.push_back(inst.getResult(argNum));
    wire->value = hw::StructCreateOp::create(builder, structTy, fields);
  }

  // File each port's wires under the edge it forms: a manager is the producer
  // for its result's (single) use; a subordinate is the consumer for its
  // upstream operand.
  for (auto [portOp, wires] : llvm::zip(ports, allWires)) {
    if (auto mgr = dyn_cast<ManagerPortOp>(portOp))
      edges[&*mgr.getPort().use_begin()].producer = std::move(wires);
    else
      edges[&portOp->getOpOperand(0)].consumer = std::move(wires);
  }
  return success();
}

LogicalResult NetworkLowering::run() {
  if (succeeded(lowerNetwork()))
    return success();
  // Avoid spurious backedge errors
  bb.abandon();
  return failure();
}

LogicalResult NetworkLowering::lowerNetwork() {
  WalkResult xbarWalk = module.walk([&](XbarOp xbar) {
    xbar.emitError("xbar lowering is not yet implemented");
    return WalkResult::interrupt();
  });
  if (xbarWalk.wasInterrupted())
    return failure();

  SmallVector<NodeOp> nodes(module.getOps<NodeOp>());
  for (NodeOp node : nodes)
    if (failed(lowerNode(node)))
      return failure();

  // Connect every edge's producer side to its consumer side.
  for (auto &[operand, edge] : edges) {
    assert(!edge.producer.empty() && !edge.consumer.empty() &&
           "both endpoints of an AXI edge must be lowered");
    connectPorts(edge.producer, edge.consumer);
  }

  // Erase the abstract ops. Repeatedly drop those with no remaining uses; the
  // SSA DAG guarantees each pass removes at least the current sinks.
  SmallVector<Operation *> dead;
  module.walk([&](Operation *op) {
    if (isa_and_nonnull<AXI4Dialect>(op->getDialect()))
      dead.push_back(op);
  });
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *&op : dead)
      if (op && op->use_empty()) {
        op->erase();
        op = nullptr;
        changed = true;
      }
  }

  return success();
}

struct AXI4ToHWPass : public circt::impl::AXI4ToHWBase<AXI4ToHWPass> {
  void runOnOperation() override;
};

} // namespace

void AXI4ToHWPass::runOnOperation() {
  if (failed(checkNetwork(getOperation())))
    return signalPassFailure();
  if (failed(NetworkLowering(getOperation()).run()))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createAXI4ToHWPass() {
  return std::make_unique<AXI4ToHWPass>();
}
