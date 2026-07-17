//===- AXI4ToHW.cpp - Translate AXI4 into HW ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers abstract networks in the AXI4 dialect to HW, wiring each node together
// according to the network's SSA edges.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AXI4ToHW.h"
#include "AXI4ToHWInternals.h"
#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/PortImplementation.h"
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
using namespace circt::AXI4ToHW;

namespace circt {
namespace AXI4ToHW {
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
    fields = {field("id", port.getWriteIdWidth()),
              field("addr", port.getAddressWidth()),
              field("len", kLenWidth),
              field("size", kSizeWidth),
              field("burst", kBurstWidth),
              field("lock", kLockWidth),
              field("cache", kCacheWidth),
              field("prot", kProtWidth),
              field("qos", kQosWidth),
              field("region", kRegionWidth),
              field("user", port.getUserWidth())};
    break;
  case AXI4Channel::AR:
    fields = {field("id", port.getReadIdWidth()),
              field("addr", port.getAddressWidth()),
              field("len", kLenWidth),
              field("size", kSizeWidth),
              field("burst", kBurstWidth),
              field("lock", kLockWidth),
              field("cache", kCacheWidth),
              field("prot", kProtWidth),
              field("qos", kQosWidth),
              field("region", kRegionWidth),
              field("user", port.getUserWidth())};
    break;
  case AXI4Channel::W:
    fields = {field("data", port.getDataWidth()),
              field("strb", port.getDataWidth() / 8), field("last", kLastWidth),
              field("user", port.getUserWidth())};
    break;
  case AXI4Channel::B:
    fields = {field("id", port.getWriteIdWidth()), field("resp", kRespWidth),
              field("user", port.getUserWidth())};
    break;
  case AXI4Channel::R:
    fields = {field("id", port.getReadIdWidth()),
              field("data", port.getDataWidth()), field("resp", kRespWidth),
              field("last", kLastWidth), field("user", port.getUserWidth())};
    break;
  }
  return hw::StructType::get(ctx, fields);
}

/// Build channel-split ports (payload, valid, ready) for a single !axi4.port.
void buildChannelPortList(MLIRContext *ctx, PortType portType, bool isManager,
                          StringRef prefix, SmallVectorImpl<PortInfo> &ports) {
  Type i1 = IntegerType::get(ctx, 1);
  auto add = [&](const Twine &name, Type t, ModulePort::Direction d) {
    ports.push_back(PortInfo{{StringAttr::get(ctx, name.str()), t, d}});
  };
  for (const ChannelInfo &ci : kChannelInfos) {
    bool payloadDrivenByModule = (isManager == ci.isRequest);
    ModulePort::Direction fwd =
        payloadDrivenByModule ? ModulePort::Output : ModulePort::Input;
    ModulePort::Direction rev =
        payloadDrivenByModule ? ModulePort::Input : ModulePort::Output;
    std::string base = (prefix + ci.token).str();
    add(base, getChannelPayloadType(portType, ci.channel), fwd);
    add(base + "_valid", i1, fwd);
    add(base + "_ready", i1, rev);
  }
}

/// Get the exclusive end of the window - returns nullopt if window does not fit
/// the `addrW`-bit address space.
std::optional<uint64_t> windowEnd(uint64_t base, uint64_t size,
                                  unsigned addrW) {
  uint64_t end = base + size;
  bool overflow = end < base;
  if (addrW >= 64) {
    // Fits unless the sum passes 2^64.
    if (!overflow)
      return end;
    return end == 0 ? std::optional<uint64_t>(0) : std::nullopt;
  }
  uint64_t top = 1ull << addrW;
  if (overflow || base >= top || end > top)
    return std::nullopt;
  return end == top ? 0 : end;
}

} // namespace AXI4ToHW
} // namespace circt

namespace {

//===----------------------------------------------------------------------===//
// Pre-pass rejections
//===----------------------------------------------------------------------===//

LogicalResult checkNetwork(ModuleOp module) {
  bool failed = false;

  // The whole network is lowered into one block; a split-region network would
  // otherwise emit instances that can't reach their peers.
  Region *commonRegion = nullptr;
  auto checkRegion = [&](Operation *op) {
    if (!commonRegion)
      commonRegion = op->getParentRegion();
    else if (op->getParentRegion() != commonRegion) {
      op->emitError("all axi4 network operations must be in the same region");
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
      op->emitError("only the 'port_wires' port_mapping is supported");
      failed = true;
    }
  };

  // Access windows must fit the port's address space; otherwise the derived
  // routing rule would wrap and silently mis-route.
  auto checkWindows = [&](Operation *op, auto access, unsigned addrW) {
    for (auto win : access.template getAsRange<WindowAttr>()) {
      uint64_t base = win.getBase(), size = win.getSize();
      if (!windowEnd(base, size, addrW)) {
        op->emitError("access window [base ")
            << base << ", size " << size << ") does not fit the " << addrW
            << "-bit address space";
        failed = true;
      }
    }
  };

  // The (clock, reset) domain of the face on `def` that produces a port value.
  auto producerDomain = [](Operation *def) -> std::pair<Value, Value> {
    if (auto mgr = dyn_cast_or_null<ManagerPortOp>(def))
      return {mgr.getClock(), mgr.getReset()};
    if (auto xbar = dyn_cast_or_null<XbarOp>(def))
      return {xbar.getClock(), xbar.getReset()};
    if (auto cut = dyn_cast_or_null<CutOp>(def))
      return {cut.getClock(), cut.getReset()};
    if (auto cdc = dyn_cast_or_null<CDCOp>(def))
      return {cdc.getDownstreamClock(), cdc.getDownstreamReset()};
    return {nullptr, nullptr};
  };

  // Every port operand of `consumer` must come from a producer in the same
  // clock/reset domain. A cdc is the only op whose two faces may differ, so a
  // mismatch here is a domain crossing that skips an axi4.cdc.
  auto checkEdgesInto = [&](Operation *consumer, Value clock, Value reset) {
    for (Value operand : consumer->getOperands()) {
      if (!isa<PortType>(operand.getType()))
        continue;
      auto [prodClock, prodReset] = producerDomain(operand.getDefiningOp());
      if ((prodClock && prodClock != clock) ||
          (prodReset && prodReset != reset)) {
        consumer->emitError("clock/reset domain crossing without an axi4.cdc; "
                            "insert an axi4.cdc to cross domains");
        failed = true;
      }
    }
  };

  module.walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<ManagerPortOp>([&](ManagerPortOp mgr) {
          checkRegion(mgr);
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
          checkWindows(
              mgr, mgr.getAccess(),
              cast<PortType>(mgr.getPort().getType()).getAddressWidth());
          checkMapping(mgr, mgr.getPortMapping());
        })
        .Case<SubordinatePortOp>([&](SubordinatePortOp sub) {
          checkRegion(sub);
          if (!sub.getNode()) {
            sub.emitError("nodeless ports are not yet supported");
            failed = true;
            return;
          }
          checkWindows(
              sub, sub.getAccess(),
              cast<PortType>(sub.getUpstream().getType()).getAddressWidth());
          checkMapping(sub, sub.getPortMapping());
          checkEdgesInto(sub, sub.getClock(), sub.getReset());
        })
        .Case<XbarOp>([&](XbarOp xbar) {
          checkRegion(xbar);
          checkUsers(xbar);
          if (checkXbarSupported(xbar).failed())
            failed = true;
          checkEdgesInto(xbar, xbar.getClock(), xbar.getReset());
        })
        .Case<CutOp>([&](CutOp cut) {
          checkRegion(cut);
          checkUsers(cut);
          // The verifier already caps the result at one use; a cut with none
          // has no downstream port to register into.
          if (cut.getDownstream().use_empty()) {
            cut.emitError("axi4.cut result must feed a downstream port");
            failed = true;
          }
          checkEdgesInto(cut, cut.getClock(), cut.getReset());
        })
        .Case<CDCOp>([&](CDCOp cdc) {
          checkRegion(cdc);
          checkUsers(cdc);
          // As for a cut, the verifier caps the result at one use; a cdc with
          // none has no downstream port to cross into.
          if (cdc.getDownstream().use_empty()) {
            cdc.emitError("axi4.cdc result must feed a downstream port");
            failed = true;
          }
          // The cdc bridges domains, so its upstream face is checked against
          // the source domain here; its downstream face is checked at whatever
          // consumes its result.
          checkEdgesInto(cdc, cdc.getUpstreamClock(), cdc.getUpstreamReset());
        })
        .Case<NodeOp>([&](NodeOp node) {
          checkRegion(node);
          checkUsers(node);
        });
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
/// ports. `FromModule` binds a module output, `ToModule` drives a module input.
/// Holds the shared port lookup and scalar wiring; subclasses map each
/// channel's payload onto the target module's ports.
class MappingLowerer {
public:
  MappingLowerer(Operation *portOp, StringRef notFoundSuffix,
                 DenseMap<StringRef, PortInfo> &portByName,
                 DenseMap<StringRef, Value> &inputByName,
                 SmallVectorImpl<std::pair<Wire *, StringRef>> &outs)
      : portOp(portOp), notFoundSuffix(notFoundSuffix.str()),
        portByName(portByName), inputByName(inputByName), outs(outs) {}
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

protected:
  std::optional<PortInfo> lookup(const Twine &nameT,
                                 ModulePort::Direction dir) {
    std::string name = nameT.str();
    auto it = portByName.find(name);
    if (it == portByName.end()) {
      portOp->emitError("referenced module has no port '")
          << name << "'" << notFoundSuffix;
      return std::nullopt;
    }
    if (it->second.dir != dir) {
      portOp->emitError("module port '") << name << "' has the wrong direction";
      return std::nullopt;
    }
    return it->second;
  }

  /// Bind a single module output port (whole value) to `wire`.
  LogicalResult registerScalarFromModule(const Twine &name, Wire &wire) {
    auto port = lookup(name, ModulePort::Output);
    if (!port)
      return failure();
    outs.emplace_back(&wire, port->name.getValue());
    return success();
  }

  /// Drive a single module input port with `wire`'s value.
  LogicalResult registerScalarToModule(const Twine &name, Wire &wire) {
    auto port = lookup(name, ModulePort::Input);
    if (!port)
      return failure();
    inputByName[port->name.getValue()] = wire.value;
    return success();
  }

  /// Op used for diagnostics.
  Operation *portOp;
  /// Appended to the "no port" diagnostic for mapping-specific context.
  std::string notFoundSuffix;
  /// Module ports by name.
  DenseMap<StringRef, PortInfo> &portByName;
  /// Instance input drivers.
  DenseMap<StringRef, Value> &inputByName;
  /// Scalar outputs resolved after instantiation.
  SmallVectorImpl<std::pair<Wire *, StringRef>> &outs;
};

/// Lowers a port group using the flat-scalar `port_wires` mapping: each channel
/// payload is split into one module port per field.
class PortWiresMappingLowerer : public MappingLowerer {
public:
  PortWiresMappingLowerer(
      Operation *portOp, bool isManager, StringRef name,
      DenseMap<StringRef, PortInfo> &portByName,
      DenseMap<StringRef, Value> &inputByName,
      SmallVectorImpl<std::pair<Wire *, StringRef>> &outs,
      SmallVectorImpl<std::tuple<Wire *, hw::StructType,
                                 SmallVector<StringRef>>> &outStructs,
      ImplicitLocOpBuilder &builder)
      : MappingLowerer(portOp, " required by the port_wires mapping",
                       portByName, inputByName, outs),
        base(((isManager ? "m_axi_" : "s_axi_") + name + "_").str()),
        outStructs(outStructs), builder(builder) {}

  LogicalResult registerPayloadFromModule(const ChannelInfo &ci,
                                          hw::StructType payloadTy,
                                          Wire &payload) override {
    std::string cbase = getChannelBase(ci);
    SmallVector<StringRef> fieldNames;
    for (auto &field : payloadTy.getElements()) {
      auto port = lookup(cbase + field.name.getValue(), ModulePort::Output);
      if (!port)
        return failure();
      fieldNames.push_back(port->name.getValue());
    }
    outStructs.emplace_back(&payload, payloadTy, std::move(fieldNames));
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
  std::string getChannelBase(const ChannelInfo &ci) const {
    return base + ci.token.str();
  }

  /// `m_axi_`/`s_axi_` port name prefix.
  std::string base;
  /// Payload outputs assembled after instantiation.
  SmallVectorImpl<std::tuple<Wire *, hw::StructType, SmallVector<StringRef>>>
      &outStructs;
  /// Insertion point for helper ops.
  ImplicitLocOpBuilder &builder;
};

/// Lowers a port group using the struct-grouped channel-split convention: each
/// channel is one whole-struct payload port plus scalar `valid`/`ready` ports,
/// named `<prefix><token>`/`_valid`/`_ready`. Used for the xbar wrapper, whose
/// interface mirrors the internal `ChannelWires` form

class ChannelPortsMappingLowerer : public MappingLowerer {
public:
  ChannelPortsMappingLowerer(
      Operation *portOp, StringRef prefix,
      DenseMap<StringRef, PortInfo> &portByName,
      DenseMap<StringRef, Value> &inputByName,
      SmallVectorImpl<std::pair<Wire *, StringRef>> &outs)
      : MappingLowerer(portOp, "", portByName, inputByName, outs),
        prefix(prefix.str()) {}

  // The module port already is the whole struct; no per-field split/reassembly.
  LogicalResult registerPayloadFromModule(const ChannelInfo &ci, hw::StructType,
                                          Wire &payload) override {
    return registerScalarFromModule(getChannelBase(ci), payload);
  }
  LogicalResult registerPayloadToModule(const ChannelInfo &ci, hw::StructType,
                                        Wire &payload) override {
    return registerScalarToModule(getChannelBase(ci), payload);
  }
  LogicalResult registerValidFromModule(const ChannelInfo &ci,
                                        Wire &valid) override {
    return registerScalarFromModule(getChannelBase(ci) + "_valid", valid);
  }
  LogicalResult registerValidToModule(const ChannelInfo &ci,
                                      Wire &valid) override {
    return registerScalarToModule(getChannelBase(ci) + "_valid", valid);
  }
  LogicalResult registerReadyFromModule(const ChannelInfo &ci,
                                        Wire &ready) override {
    return registerScalarFromModule(getChannelBase(ci) + "_ready", ready);
  }
  LogicalResult registerReadyToModule(const ChannelInfo &ci,
                                      Wire &ready) override {
    return registerScalarToModule(getChannelBase(ci) + "_ready", ready);
  }

private:
  std::string getChannelBase(const ChannelInfo &ci) const {
    return prefix + ci.token.str();
  }

  /// Per-face port name prefix (e.g. `sub0_`/`mgr0_`).
  std::string prefix;
};

/// Populate `wires` for one port group, walking the five channels and placing
/// backedges by direction; `mapping` supplies the mapping-specific port wiring.
LogicalResult populatePortGroup(bool isManager, PortType portType,
                                ImplicitLocOpBuilder &builder,
                                BackedgeBuilder &bb, PortWires &wires,
                                MappingLowerer &mapping) {
  Type i1 = builder.getI1Type();

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

} // namespace

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

// Materialize resets as i1, mirroring materializeClock.
Value NetworkLowering::materializeReset(Value axiReset, Type portType) {
  if (!portType.isInteger(1))
    return {};
  Value &reset = resetCache[axiReset];
  if (!reset)
    reset = UnrealizedConversionCastOp::create(builder, portType, axiReset)
                .getResult(0);
  return reset;
}

LogicalResult NetworkLowering::buildInstance(
    Operation *diag, hw::HWModuleLike moduleOp, StringRef instanceName,
    ArrayRef<PortGroupSpec> specs, SmallVectorImpl<PortWires> &wiresOut) {
  DenseMap<StringRef, PortInfo> portByName;
  for (auto &p : moduleOp.getPortList())
    portByName[p.getName()] = p;

  DenseMap<StringRef, Value> inputByName;
  SmallVector<std::pair<Wire *, StringRef>> outs;
  SmallVector<std::tuple<Wire *, hw::StructType, SmallVector<StringRef>>>
      outStructs;

  // Reserve so the deferred output pointers below stay valid:
  // `outs`/`outStructs` hold Wire pointers into these elements until we resolve
  // them post-instance.
  wiresOut.reserve(specs.size());
  for (const PortGroupSpec &spec : specs) {
    wiresOut.emplace_back();
    std::unique_ptr<MappingLowerer> lowerer;
    switch (spec.kind) {
    case MappingKind::PortWires:
      lowerer = std::make_unique<PortWiresMappingLowerer>(
          spec.defOp, spec.isManager, spec.name, portByName, inputByName, outs,
          outStructs, builder);
      break;
    case MappingKind::ChannelPorts:
      lowerer = std::make_unique<ChannelPortsMappingLowerer>(
          spec.defOp, spec.name, portByName, inputByName, outs);
      break;
    }
    if (failed(populatePortGroup(spec.isManager, spec.portType, builder, bb,
                                 wiresOut.back(), *lowerer)))
      return failure();

    auto clockPort = portByName.find(spec.clockPort);
    if (clockPort == portByName.end())
      return spec.defOp->emitError("referenced module has no clock port '")
             << spec.clockPort << "'";
    Value clock = materializeClock(spec.axiClock, clockPort->second.type);
    if (!clock)
      return spec.defOp->emitError("unsupported clock port type");
    inputByName[clockPort->second.name.getValue()] = clock;

    auto resetPort = portByName.find(spec.resetPort);
    if (resetPort == portByName.end())
      return spec.defOp->emitError("referenced module has no reset port '")
             << spec.resetPort << "'";
    Value reset = materializeReset(spec.axiReset, resetPort->second.type);
    if (!reset)
      return spec.defOp->emitError("unsupported reset port type");
    inputByName[resetPort->second.name.getValue()] = reset;
  }

  // Assemble instance inputs in the module's input-port order.
  SmallVector<Value> inputs;
  for (auto &p : moduleOp.getPortList()) {
    if (!p.isInput())
      continue;
    auto it = inputByName.find(p.getName());
    if (it == inputByName.end())
      return diag->emitError("module input port '")
             << p.getName() << "' is not driven by any AXI4 port";
    inputs.push_back(it->second);
  }

  auto inst = hw::InstanceOp::create(
      builder, moduleOp, builder.getStringAttr(instanceName), inputs);

  // Instance results correspond to output ports in port-list order; map each
  // output port name to its result index. (Resolving by name rather than
  // PortInfo::argNum keeps this correct across module kinds - some HWModuleLike
  // ops populate argNum with the combined port index rather than the
  // direction-relative one.)
  DenseMap<StringRef, unsigned> outputResult;
  unsigned resultIdx = 0;
  for (auto &p : moduleOp.getPortList())
    if (p.isOutput())
      outputResult[p.getName()] = resultIdx++;

  // Resolve deferred module outputs into real driving values.
  for (auto &[wire, name] : outs)
    wire->value = inst.getResult(outputResult.at(name));
  for (auto &[wire, structTy, fieldNames] : outStructs) {
    SmallVector<Value> fields;
    for (StringRef fn : fieldNames)
      fields.push_back(inst.getResult(outputResult.at(fn)));
    wire->value = hw::StructCreateOp::create(builder, structTy, fields);
  }
  return success();
}

LogicalResult NetworkLowering::lowerNode(NodeOp node) {
  SmallVector<Operation *> ports(node.getNode().getUsers());
  if (ports.empty())
    return success();

  // The NodeOp verifier guarantees the symbol resolves to an hw.module.
  auto moduleOp = module.lookupSymbol<hw::HWModuleLike>(node.getModuleAttr());
  assert(moduleOp && "node symbol does not map to module");

  builder.setLoc(node.getLoc());

  // One interface per attached port.
  SmallVector<PortGroupSpec> specs;
  for (Operation *portOp : ports) {
    Value axiClock, axiReset;
    PortType portType;
    AXI4PortMappingAttrInterface mapping;
    if (auto mgr = dyn_cast<ManagerPortOp>(portOp)) {
      axiClock = mgr.getClock();
      axiReset = mgr.getReset();
      portType = cast<PortType>(mgr.getPort().getType());
      mapping = *mgr.getPortMapping();
    } else {
      auto sub = cast<SubordinatePortOp>(portOp);
      axiClock = sub.getClock();
      axiReset = sub.getReset();
      portType = cast<PortType>(sub.getUpstream().getType());
      mapping = *sub.getPortMapping();
    }

    auto portWires = dyn_cast<PortWiresAttr>(mapping);
    if (!portWires)
      return portOp->emitError(
          "only the 'port_wires' port_mapping is supported");
    specs.push_back({portOp, isa<ManagerPortOp>(portOp), portType,
                     MappingKind::PortWires, portWires.getName().str(),
                     axiClock, mapping.getClockPort().str(), axiReset,
                     mapping.getResetPort().str()});
  }

  SmallVector<PortWires, 0> allWires;
  std::string instName =
      (node.getModule() + "_" + Twine(instanceCounter++)).str();
  if (failed(buildInstance(node, moduleOp, instName, specs, allWires)))
    return failure();

  // File each port's wires under the edge it forms: a manager is the producer
  // for its result's (single) use; a subordinate is the consumer for its
  // upstream operand.
  for (auto [portOp, wires] : llvm::zip(ports, allWires)) {
    if (auto mgr = dyn_cast<ManagerPortOp>(portOp))
      edges[&*mgr.getPort().use_begin()].producer = wires;
    else
      edges[&portOp->getOpOperand(0)].consumer = wires;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

LogicalResult NetworkLowering::run() {
  if (succeeded(lowerNetwork()))
    return success();
  // Avoid spurious backedge errors
  bb.abandon();
  return failure();
}

LogicalResult NetworkLowering::lowerNetwork() {
  // The network may sit at the top level or inside an enclosing hw.module; find
  // its ops wherever they are and emit the lowered design in the same block
  SmallVector<NodeOp> nodes;
  SmallVector<XbarOp> xbars;
  SmallVector<CutOp> cuts;
  SmallVector<CDCOp> cdcs;
  module.walk([&](Operation *op) {
    if (auto node = dyn_cast<NodeOp>(op))
      nodes.push_back(node);
    else if (auto xbar = dyn_cast<XbarOp>(op))
      xbars.push_back(xbar);
    else if (auto cut = dyn_cast<CutOp>(op))
      cuts.push_back(cut);
    else if (auto cdc = dyn_cast<CDCOp>(op))
      cdcs.push_back(cdc);
  });

  Block *netBlock = nullptr;
  if (!nodes.empty())
    netBlock = nodes.front()->getBlock();
  else if (!xbars.empty())
    netBlock = xbars.front()->getBlock();
  else if (!cuts.empty())
    netBlock = cuts.front()->getBlock();
  else if (!cdcs.empty())
    netBlock = cdcs.front()->getBlock();
  if (netBlock) {
    if (!netBlock->empty() &&
        netBlock->back().hasTrait<OpTrait::IsTerminator>())
      builder.setInsertionPoint(&netBlock->back());
    else
      builder.setInsertionPointToEnd(netBlock);
  }

  for (NodeOp node : nodes)
    if (failed(lowerNode(node)))
      return failure();

  for (XbarOp xbar : xbars)
    if (failed(lowerXbar(xbar)))
      return failure();

  for (CutOp cut : cuts)
    if (failed(lowerCut(cut)))
      return failure();

  for (CDCOp cdc : cdcs)
    if (failed(lowerCdc(cdc)))
      return failure();

  // Connect every edge's producer side to its consumer side.
  for (auto &[operand, edge] : edges) {
    assert(edge.producer && edge.consumer &&
           "both endpoints of an AXI edge must be lowered");
    connectPorts(*edge.producer, *edge.consumer);
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

namespace {
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
