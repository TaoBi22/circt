//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers an AXI4 network specification to a concrete RTL description.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AXI4ToHW.h"
#include "AXI4ToHWInternals.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_AXI4TOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;
using namespace circt::AXI4ToHW;

// axi4.ports split into fifteen signals - a valid, a ready and a payload for
// each channel

namespace {
/// One of the signals an `!axi4.port` explodes into.
struct SignalInfo {
  std::string suffix;
  Type type;
  bool managerDrives;
};
} // namespace

/// The signals `port` explodes into. A channel's ready travels against its
/// payload, so the end that drives one receives the other.
static SmallVector<SignalInfo, 15> portSignals(PortType port) {
  Type i1 = IntegerType::get(port.getContext(), 1);
  SmallVector<SignalInfo, 15> signals;
  for (const ChannelInfo &info : kChannels) {
    Type payload = getChannelPayloadType(port, info.channel);
    signals.push_back(
        {(Twine("_") + info.name).str(), payload, info.isRequest});
    signals.push_back(
        {(Twine("_") + info.name + "valid").str(), i1, info.isRequest});
    signals.push_back(
        {(Twine("_") + info.name + "ready").str(), i1, !info.isRequest});
  }
  return signals;
}

/// The types of the signals `port` explodes into that one end drives, in AXI4
/// order - the order the bridge ops take and return them in.
static SmallVector<Type> signalTypes(PortType port, bool managerDriven) {
  SmallVector<Type> types;
  for (const SignalInfo &signal : portSignals(port))
    if (signal.managerDrives == managerDriven)
      types.push_back(signal.type);
  return types;
}

/// The types of `ports`, in order.
static SmallVector<Type> portTypes(ArrayRef<hw::PortInfo> ports) {
  return llvm::map_to_vector(
      ports, [](const hw::PortInfo &port) { return port.type; });
}

//===----------------------------------------------------------------------===//
// Request and response structs
//===----------------------------------------------------------------------===//

/// The width of the `atop` field PULP's AW channel carries and the dialect does
/// not model.
static constexpr unsigned kAtopWidth = 6;

namespace {
/// A field of PULP's `axi_req_t` or `axi_resp_t`: its name, the index its
/// signal takes in AXI4 order, and the channel whose payload it carries, if it
/// carries one.
struct StructField {
  StringLiteral name;
  unsigned index;
  std::optional<AXI4Channel> payload;
};
} // namespace

/// The fields of `axi_req_t`, carrying the signals a manager drives.
static constexpr StructField kRequestFields[] = {
    {StringLiteral("aw"), 0, AXI4Channel::AW},
    {StringLiteral("aw_valid"), 1, std::nullopt},
    {StringLiteral("w"), 2, AXI4Channel::W},
    {StringLiteral("w_valid"), 3, std::nullopt},
    {StringLiteral("b_ready"), 4, std::nullopt},
    {StringLiteral("ar"), 5, AXI4Channel::AR},
    {StringLiteral("ar_valid"), 6, std::nullopt},
    {StringLiteral("r_ready"), 7, std::nullopt}};

/// The fields of `axi_resp_t`, carrying the signals a subordinate drives. PULP
/// orders them differently to the AXI4 signal order.
static constexpr StructField kResponseFields[] = {
    {StringLiteral("aw_ready"), 0, std::nullopt},
    {StringLiteral("ar_ready"), 4, std::nullopt},
    {StringLiteral("w_ready"), 1, std::nullopt},
    {StringLiteral("b_valid"), 3, std::nullopt},
    {StringLiteral("b"), 2, AXI4Channel::B},
    {StringLiteral("r_valid"), 6, std::nullopt},
    {StringLiteral("r"), 5, AXI4Channel::R}};

/// The payload struct PULP's typedefs build for `channel`: the dialect's
/// payload plus the fields PULP always carries - AW's `atop`, and a `user` of
/// at least one bit.
static hw::StructType pulpPayloadType(PortType port, AXI4Channel channel) {
  MLIRContext *context = port.getContext();
  SmallVector<hw::StructType::FieldInfo> fields;
  for (const auto &field : getChannelPayloadType(port, channel).getElements()) {
    // `user` ends every channel, and `atop` sits just ahead of it.
    if (field.name.getValue() != "user") {
      fields.push_back(field);
      continue;
    }
    if (channel == AXI4Channel::AW)
      fields.push_back({StringAttr::get(context, "atop"),
                        IntegerType::get(context, kAtopWidth)});
    fields.push_back(
        {field.name, IntegerType::get(context, pulpUserWidth(port))});
  }
  return hw::StructType::get(context, fields);
}

/// The struct `layout` describes for `port`.
static hw::StructType structType(PortType port, ArrayRef<StructField> layout) {
  MLIRContext *context = port.getContext();
  return hw::StructType::get(
      context, llvm::map_to_vector(layout, [&](const StructField &field) {
        Type type = field.payload ? Type(pulpPayloadType(port, *field.payload))
                                  : IntegerType::get(context, 1);
        return hw::StructType::FieldInfo{StringAttr::get(context, field.name),
                                         type};
      }));
}

/// The fields of the struct `value`, by name.
static llvm::StringMap<Value> explodeFields(ImplicitLocOpBuilder &b,
                                            Value value) {
  auto type = cast<hw::StructType>(value.getType());
  auto exploded = hw::StructExplodeOp::create(b, value);
  llvm::StringMap<Value> fields;
  for (auto [field, result] :
       llvm::zip_equal(type.getElements(), exploded.getResults()))
    fields[field.name.getValue()] = result;
  return fields;
}

/// Repack `payload` into PULP's layout, zeroing the fields PULP carries that
/// the dialect does not.
static Value toPulpPayload(ImplicitLocOpBuilder &b, PortType port,
                           AXI4Channel channel, Value payload) {
  hw::StructType type = pulpPayloadType(port, channel);
  if (type == payload.getType())
    return payload;
  llvm::StringMap<Value> fields = explodeFields(b, payload);
  auto values = llvm::map_to_vector(
      type.getElements(), [&](const hw::StructType::FieldInfo &field) -> Value {
        StringRef name = field.name.getValue();
        // `atop` has no signal behind it, and neither has `user` on a port
        // carrying none.
        if (name == "atop" || (name == "user" && port.getUserWidth() == 0))
          return hw::ConstantOp::create(
              b, APInt::getZero(hw::getBitWidth(field.type)));
        return fields.lookup(name);
      });
  return hw::StructCreateOp::create(b, type, values);
}

/// Repack `payload` from PULP's layout into the dialect's, dropping the fields
/// the dialect does not carry.
static Value fromPulpPayload(ImplicitLocOpBuilder &b, PortType port,
                             AXI4Channel channel, Value payload) {
  hw::StructType type = getChannelPayloadType(port, channel);
  if (type == payload.getType())
    return payload;
  llvm::StringMap<Value> fields = explodeFields(b, payload);
  auto values = llvm::map_to_vector(
      type.getElements(), [&](const hw::StructType::FieldInfo &field) -> Value {
        StringRef name = field.name.getValue();
        if (name == "user" && port.getUserWidth() == 0)
          return hw::ConstantOp::create(b, APInt::getZero(0));
        return fields.lookup(name);
      });
  return hw::StructCreateOp::create(b, type, values);
}

/// The signals the struct `value` of `layout` carries, in AXI4 order.
static SmallVector<Value> explodeStruct(ImplicitLocOpBuilder &b, PortType port,
                                        ArrayRef<StructField> layout,
                                        Value value) {
  llvm::StringMap<Value> fields = explodeFields(b, value);
  SmallVector<Value> signals(layout.size());
  for (const StructField &field : layout) {
    Value signal = fields.lookup(field.name);
    signals[field.index] =
        field.payload ? fromPulpPayload(b, port, *field.payload, signal)
                      : signal;
  }
  return signals;
}

/// The struct of `layout` carrying `signals`, which arrive in AXI4 order.
static Value packStruct(ImplicitLocOpBuilder &b, PortType port,
                        ArrayRef<StructField> layout, ValueRange signals) {
  auto fields =
      llvm::map_to_vector(layout, [&](const StructField &field) -> Value {
        Value signal = signals[field.index];
        return field.payload ? toPulpPayload(b, port, *field.payload, signal)
                             : signal;
      });
  return hw::StructCreateOp::create(b, structType(port, layout), fields);
}

//===----------------------------------------------------------------------===//
// Components
//===----------------------------------------------------------------------===//

namespace {
/// An AXI4 op lowered to an instance of an external module of its shape.
struct Component {
  Operation *op;
  /// The name of the external module implementing it
  std::string moduleName;
  /// The name its instances take, suffixed to count them within a module
  StringRef instanceName;
  /// The ports it takes ahead of its upstream ones, in order, and what drives
  /// them - its clock and reset, and whatever else it is fed
  SmallVector<std::pair<StringRef, Value>> inputs;
  /// The names its results take, for a component whose results are not
  /// downstream ports
  SmallVector<StringRef> outputs;
};
} // namespace

/// The `!axi4.port` operands of `op`, which arrive from upstream.
static SmallVector<Value> upstreamPorts(Operation *op) {
  SmallVector<Value> ports;
  for (Value operand : op->getOperands())
    if (isa<PortType>(operand.getType()))
      ports.push_back(operand);
  return ports;
}

/// The suffix naming the shape of a component carrying `port`.
static std::string portShape(PortType port) {
  return ("a" + Twine(port.getAddrWidth()) + "_d" + Twine(port.getDataWidth()) +
          "_i" + Twine(port.getWriteIdWidth()))
      .str();
}

/// The component `op` lowers to, or nothing if it is not one.
static std::optional<Component> getComponent(Operation *op) {
  return TypeSwitch<Operation *, std::optional<Component>>(op)
      .Case<XbarOp>([](XbarOp xbar) {
        auto upstream = cast<PortType>(xbar.getUpstream().front().getType());
        auto downstream =
            cast<PortType>(xbar.getDownstream().front().getType());
        return Component{
            xbar,
            ("axi_xbar_" + Twine(xbar.getUpstream().size()) + "u" +
             Twine(xbar.getDownstream().size()) + "d_" + portShape(upstream) +
             "_o" + Twine(downstream.getWriteIdWidth()))
                .str(),
            "xbar",
            {{"clk_i", xbar.getClock()}, {"rst_ni", xbar.getReset()}}};
      })
      .Case<CutOp>([](CutOp cut) {
        auto port = cast<PortType>(cut.getUpstream().getType());
        return Component{
            cut,
            "axi_cut_" + portShape(port),
            "cut",
            {{"clk_i", cut.getClock()}, {"rst_ni", cut.getReset()}}};
      })
      .Case<CDCOp>([](CDCOp cdc) {
        auto port = cast<PortType>(cdc.getUpstream().getType());
        return Component{cdc,
                         "axi_cdc_" + portShape(port),
                         "cdc",
                         {{"src_clk_i", cdc.getUpstreamClock()},
                          {"dst_clk_i", cdc.getDownstreamClock()},
                          {"rst_ni", cdc.getReset()}}};
      })
      .Case<DWConverterOp>([](DWConverterOp converter) {
        auto upstream = cast<PortType>(converter.getUpstream().getType());
        auto downstream = cast<PortType>(converter.getDownstream().getType());
        return Component{converter,
                         ("axi_dw_converter_a" +
                          Twine(upstream.getAddrWidth()) + "_d" +
                          Twine(upstream.getDataWidth()) + "to" +
                          Twine(downstream.getDataWidth()) + "_i" +
                          Twine(upstream.getWriteIdWidth()))
                             .str(),
                         "dw_converter",
                         {{"clk_i", converter.getClock()},
                          {"rst_ni", converter.getReset()}}};
      })
      .Case<IWConverterOp>([](IWConverterOp converter) {
        auto upstream = cast<PortType>(converter.getUpstream().getType());
        auto downstream = cast<PortType>(converter.getDownstream().getType());
        return Component{converter,
                         ("axi_iw_converter_" + portShape(upstream) + "to" +
                          Twine(downstream.getWriteIdWidth()))
                             .str(),
                         "iw_converter",
                         {{"clk_i", converter.getClock()},
                          {"rst_ni", converter.getReset()}}};
      })
      .Case<DemuxOp>([](DemuxOp demux) {
        auto port = cast<PortType>(demux.getUpstream().getType());
        return Component{
            demux,
            ("axi_demux_" + Twine(demux.getDownstream().size()) + "d_" +
             portShape(port))
                .str(),
            "demux",
            {{"clk_i", demux.getClock()}, {"rst_ni", demux.getReset()}}};
      })
      .Case<MuxOp>([](MuxOp mux) {
        auto upstream = cast<PortType>(mux.getUpstream().front().getType());
        auto downstream = cast<PortType>(mux.getDownstream().getType());
        return Component{
            mux,
            ("axi_mux_" + Twine(mux.getUpstream().size()) + "u_" +
             portShape(upstream) + "_o" + Twine(downstream.getWriteIdWidth()))
                .str(),
            "mux",
            {{"clk_i", mux.getClock()}, {"rst_ni", mux.getReset()}}};
      })
      .Case<BurstSplitterOp>([](BurstSplitterOp splitter) {
        auto port = cast<PortType>(splitter.getUpstream().getType());
        return Component{
            splitter,
            "axi_burst_splitter_" + portShape(port),
            "burst_splitter",
            {{"clk_i", splitter.getClock()}, {"rst_ni", splitter.getReset()}}};
      })
      .Case<BurstUnwrapperOp>([](BurstUnwrapperOp unwrapper) {
        auto port = cast<PortType>(unwrapper.getUpstream().getType());
        return Component{unwrapper,
                         "axi_burst_unwrapper_" + portShape(port),
                         "burst_unwrapper",
                         {{"clk_i", unwrapper.getClock()},
                          {"rst_ni", unwrapper.getReset()}}};
      })
      .Case<ToMemOp>([](ToMemOp toMem) {
        auto port = cast<PortType>(toMem.getPort().getType());
        return Component{toMem,
                         "axi_to_mem_" + portShape(port),
                         "to_mem",
                         {{"clk_i", toMem.getClock()},
                          {"rst_ni", toMem.getReset()},
                          {"mem_rvalid_i", toMem.getReadValid()},
                          {"mem_rdata_i", toMem.getReadData()}},
                         {"mem_req_o", "mem_addr_o", "mem_wdata_o",
                          "mem_strb_o", "mem_we_o"}};
      })
      .Default(std::nullopt);
}

/// The ports of the module implementing `component`. A component is the
/// subordinate to its upstream managers, so upstream ports are inputs and
/// downstream ports are outputs.
static SmallVector<hw::ModulePort> componentPorts(const Component &component) {
  MLIRContext *context = component.op->getContext();
  SmallVector<hw::ModulePort> ports;
  for (auto [name, value] : component.inputs)
    ports.push_back({StringAttr::get(context, name), value.getType(),
                     hw::ModulePort::Direction::Input});
  for (auto [index, value] : llvm::enumerate(upstreamPorts(component.op)))
    ports.push_back({StringAttr::get(context, "mgr" + Twine(index)),
                     value.getType(), hw::ModulePort::Direction::Input});
  for (auto [index, value] : llvm::enumerate(component.op->getResults())) {
    std::string name = component.outputs.empty()
                           ? ("sub" + Twine(index)).str()
                           : component.outputs[index].str();
    ports.push_back({StringAttr::get(context, name), value.getType(),
                     hw::ModulePort::Direction::Output});
  }
  return ports;
}

/// Replace every component with an instance of an external module of its shape,
/// shared by components of the same kind whose ports match.
static LogicalResult lowerComponents(ModuleOp module, bool pulpMapping,
                                     DenseSet<Operation *> &componentModules) {
  SmallVector<Component> components;
  module.walk([&](Operation *op) {
    if (std::optional<Component> component = getComponent(op))
      components.push_back(std::move(*component));
  });

  DenseMap<std::pair<StringAttr, hw::ModuleType>, hw::HWModuleExternOp> shapes;
  DenseMap<std::pair<Operation *, StringRef>, unsigned> instanceCounts;
  SymbolTable symbolTable(module);
  auto b =
      ImplicitLocOpBuilder::atBlockBegin(module.getLoc(), module.getBody());
  for (const Component &component : components) {
    Operation *op = component.op;
    if (pulpMapping && failed(checkPulpSupported(op)))
      return failure();

    SmallVector<hw::ModulePort> ports = componentPorts(component);
    // Two kinds of component can share a port list, so the name - which encodes
    // the kind - is part of the shape.
    auto name = b.getStringAttr(component.moduleName);
    hw::HWModuleExternOp &shape =
        shapes[{name, hw::ModuleType::get(module.getContext(), ports)}];
    if (!shape) {
      shape = hw::HWModuleExternOp::create(
          b, name, llvm::map_to_vector(ports, [](hw::ModulePort port) {
            return hw::PortInfo{port};
          }));
      // Two shapes can want the same name, so let the symbol table unique it.
      symbolTable.insert(shape);
      componentModules.insert(shape);
      if (pulpMapping)
        attachPulpSource(b, shape, op);
    }

    SmallVector<Value> inputs = llvm::map_to_vector(
        component.inputs,
        [](const std::pair<StringRef, Value> &input) { return input.second; });
    llvm::append_range(inputs, upstreamPorts(op));
    ImplicitLocOpBuilder opBuilder(op->getLoc(), op);
    unsigned &count =
        instanceCounts[{op->getParentOp(), component.instanceName}];
    auto instance = hw::InstanceOp::create(
        opBuilder, shape,
        opBuilder.getStringAttr(component.instanceName + Twine(count++)),
        inputs);
    op->replaceAllUsesWith(instance.getResults());
    op->erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Bridges
//===----------------------------------------------------------------------===//

namespace {
/// Placeholder ('filler') clock and reset values for struct<->port ops that are
/// temporarily materialized between PortConverter invocations
class FillerDomain {
public:
  /// The filler for `block`, created on first use.
  std::pair<Value, Value> get(Block *block);
  /// Whether `value` is a placeholder, and so carries no domain of its own.
  bool isFiller(Value value) const;
  /// Erase the fillers that are no longer used (should erase all fillers at the
  /// end of the run).
  void eraseUnused();

private:
  DenseMap<Block *, std::pair<Value, Value>> fillers;
};
} // namespace

std::pair<Value, Value> FillerDomain::get(Block *block) {
  auto it = fillers.find(block);
  if (it != fillers.end())
    return it->second;

  ImplicitLocOpBuilder b(block->getParentOp()->getLoc(), block, block->begin());
  std::pair<Value, Value> filler{
      seq::ConstClockOp::create(b, seq::ClockConst::Low),
      hw::ConstantOp::create(b, APInt(1, 0))};
  fillers.insert({block, filler});
  return filler;
}

bool FillerDomain::isFiller(Value value) const {
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  auto it = fillers.find(op->getBlock());
  if (it == fillers.end())
    return false;
  return value == it->second.first || value == it->second.second;
}

void FillerDomain::eraseUnused() {
  for (auto &[clock, reset] : llvm::make_second_range(fillers)) {
    if (clock.use_empty())
      clock.getDefiningOp()->erase();
    if (reset.use_empty())
      reset.getDefiningOp()->erase();
  }
}

/// Report two conversion ops connected by a port but in different domains.
static LogicalResult emitDomainCrossing(Operation *op, Operation *other,
                                        StringRef domain) {
  auto diag = op->emitOpError()
              << "is in a different " << domain << " domain to the '"
              << other->getName().getStringRef() << "' connected to it";
  diag.attachNote(other->getLoc()) << "connected operation here";
  return failure();
}

/// Wire through and erase every back-to-back bridge (port<->structs op) pair -
/// errors if there's a domain crossing.
static LogicalResult annihilateBridges(ModuleOp module, FillerDomain &filler) {
  SmallVector<ChannelStructsToPortOp> toPortOps;
  module.walk([&](ChannelStructsToPortOp op) { toPortOps.push_back(op); });

  for (ChannelStructsToPortOp toPort : toPortOps) {
    if (!toPort.getPort().hasOneUse())
      continue;
    auto fromPort =
        dyn_cast<PortToChannelStructsOp>(*toPort.getPort().user_begin());
    if (!fromPort)
      continue;

    // Complain if we have two non-filler converters back to back in different
    // domains
    if (!filler.isFiller(toPort.getClock()) &&
        !filler.isFiller(fromPort.getClock())) {
      if (toPort.getClock() != fromPort.getClock())
        return emitDomainCrossing(fromPort, toPort, "clock");
      if (toPort.getReset() != fromPort.getReset())
        return emitDomainCrossing(fromPort, toPort, "reset");
    }

    // Pass signals directly past the adaptor pair
    for (auto [result, operand] : llvm::zip_equal(
             fromPort.getResults(), toPort.getOperands().drop_front(2)))
      result.replaceAllUsesWith(operand);
    for (auto [result, operand] :
         llvm::zip_equal(toPort.getResults().drop_front(),
                         fromPort.getOperands().drop_front(3)))
      result.replaceAllUsesWith(operand);

    fromPort.erase();
    toPort.erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Port conversion
//===----------------------------------------------------------------------===//

/// The number of outstanding writes/reads that can concurrently be in flight
/// down `port`.
static SmallVector<NamedAttribute, 2> concurrency(Builder &b, PortType port) {
  return {b.getNamedAttr("concurrent_writes",
                         b.getI32IntegerAttr(port.getOutstandingWrites())),
          b.getNamedAttr("concurrent_reads",
                         b.getI32IntegerAttr(port.getOutstandingReads()))};
}

namespace {
/// Lowers an `!axi4.port` module port to a payload, valid and ready port per
/// channel, bridged to the original port value by a channel struct op.
class AXI4PortConversion : public hw::PortConversion {
public:
  AXI4PortConversion(hw::PortConverterImpl &converter, hw::PortInfo origPort,
                     FillerDomain &filler)
      : PortConversion(converter, origPort), filler(filler),
        portType(cast<PortType>(origPort.type)) {}

protected:
  void buildInputSignals() override;
  void buildOutputSignals() override;
  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  /// Whether this module is the manager on the port, and so drives the signals
  /// a manager drives.
  bool isManager() const {
    return origPort.dir == hw::ModulePort::Direction::Output;
  }
  /// Create the exploded ports this module takes, and return their values.
  SmallVector<Value> createInputPorts();
  /// Create an output port for each value in `values` (and connect
  /// accordingly).
  void createOutputPorts(ValueRange values);
  /// The signals the new instance drives for this port, in signal order. They
  /// come back as backedges because the instance is not built yet.
  SmallVector<Value> instanceDriven(ArrayRef<Backedge> newResults);

  FillerDomain &filler;
  PortType portType;
  /// The generated ports, split by this module's own direction, in AXI signal
  /// order
  SmallVector<hw::PortInfo> inputPorts, outputPorts;
};

/// Lowers an `!axi4.port` module port to a request and a response struct in the
/// layout of PULP's `axi_req_t` and `axi_resp_t`, bridged to the original port
/// value by a channel struct op.
class AXI4StructPortConversion : public hw::PortConversion {
public:
  AXI4StructPortConversion(hw::PortConverterImpl &converter,
                           hw::PortInfo origPort, FillerDomain &filler)
      : PortConversion(converter, origPort), filler(filler),
        portType(cast<PortType>(origPort.type)) {}

protected:
  void buildInputSignals() override;
  void buildOutputSignals() override;
  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  /// The types a port bridge returns: the port, then the signals the
  /// subordinate end drives.
  SmallVector<Type> portAndResponseTypes();

  FillerDomain &filler;
  PortType portType;
  /// The generated request and response ports
  hw::PortInfo reqPort, respPort;
};

class AXI4PortConversionBuilder : public hw::PortConversionBuilder {
public:
  AXI4PortConversionBuilder(hw::PortConverterImpl &converter,
                            FillerDomain &filler, bool structPorts)
      : PortConversionBuilder(converter), filler(filler),
        structPorts(structPorts) {}

  FailureOr<std::unique_ptr<hw::PortConversion>>
  build(hw::PortInfo port) override {
    if (isa<PortType>(port.type)) {
      if (structPorts)
        return {std::make_unique<AXI4StructPortConversion>(converter, port,
                                                           filler)};
      return {std::make_unique<AXI4PortConversion>(converter, port, filler)};
    }
    return PortConversionBuilder::build(port);
  }

private:
  FillerDomain &filler;
  bool structPorts;
};
} // namespace

SmallVector<Value> AXI4PortConversion::createInputPorts() {
  SmallVector<Value> values;
  for (const SignalInfo &signal : portSignals(portType)) {
    if (signal.managerDrives == isManager())
      continue;
    hw::PortInfo &port = inputPorts.emplace_back();
    values.push_back(
        converter.createNewInput(origPort, signal.suffix, signal.type, port));
  }
  return values;
}

void AXI4PortConversion::createOutputPorts(ValueRange values) {
  for (const SignalInfo &signal : portSignals(portType)) {
    if (signal.managerDrives != isManager())
      continue;
    hw::PortInfo &port = outputPorts.emplace_back();
    Value value = values.empty() ? Value{} : values[outputPorts.size() - 1];
    converter.createNewOutput(origPort, signal.suffix, signal.type, value,
                              port);
  }
}

SmallVector<Value>
AXI4PortConversion::instanceDriven(ArrayRef<Backedge> newResults) {
  return llvm::map_to_vector(outputPorts,
                             [&](const hw::PortInfo &port) -> Value {
                               return newResults[port.argNum];
                             });
}

/// Build the corresponding ports for an !axi4.port input
void AXI4PortConversion::buildInputSignals() {
  // This hook is called when an !axi.port is an input to a module, so we're in
  // a subordinate
  SmallVector<Value> inputs = createInputPorts();

  // A module with no body (e.g. extern) can't do anything with values, so just
  // add the ports
  if (!body) {
    createOutputPorts({});
    return;
  }

  // Turn the !axi.port value that was already being digested into a set of
  // signals
  auto [clock, reset] = filler.get(body);
  SmallVector<Value> operands{clock, reset};
  llvm::append_range(operands, inputs);
  SmallVector<Type> resultTypes{portType};
  llvm::append_range(resultTypes, signalTypes(portType, isManager()));

  ImplicitLocOpBuilder b(origPort.loc, body->getTerminator());
  auto toPort = ChannelStructsToPortOp::create(b, resultTypes, operands);
  body->getArgument(origPort.argNum).replaceAllUsesWith(toPort.getPort());
  createOutputPorts(toPort.getResults().drop_front());
}

/// Build the corresponding ports for an !axi4.port output
void AXI4PortConversion::buildOutputSignals() {
  // This hook is called when an !axi.port is an output of a module, so we're in
  // a manager
  SmallVector<Value> inputs = createInputPorts();

  // A module with no body (e.g. extern) can't do anything with values, so just
  // add the ports
  if (!body) {
    createOutputPorts({});
    return;
  }

  // Turn the !axi.port value that was already being driven into a set of
  // signals
  auto [clock, reset] = filler.get(body);
  Operation *terminator = body->getTerminator();
  SmallVector<Value> operands{clock, reset,
                              terminator->getOperand(origPort.argNum)};
  llvm::append_range(operands, inputs);

  ImplicitLocOpBuilder b(origPort.loc, terminator);
  auto fromPort =
      PortToChannelStructsOp::create(b, signalTypes(portType, isManager()),
                                     operands, concurrency(b, portType));
  createOutputPorts(fromPort.getResults());
}

// Map to the newly created ports on the instances of the modified subordinate
void AXI4PortConversion::mapInputSignals(OpBuilder &b, Operation *inst,
                                         Value instValue,
                                         SmallVectorImpl<Value> &newOperands,
                                         ArrayRef<Backedge> newResults) {
  // Create a PortToChannelStructsOp to break down the port that was previously
  // an input to the instance into individual signals (valid/ready and payload
  // for each channel)
  auto [clock, reset] = filler.get(inst->getBlock());
  SmallVector<Value> operands{clock, reset, instValue};
  llvm::append_range(operands, instanceDriven(newResults));

  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  auto fromPort = PortToChannelStructsOp::create(
      builder, portTypes(inputPorts), operands, concurrency(builder, portType));

  for (auto [port, value] : llvm::zip_equal(inputPorts, fromPort.getResults()))
    newOperands[port.argNum] = value;
}

// Map to the newly created ports on the instances of the modified manager
void AXI4PortConversion::mapOutputSignals(OpBuilder &b, Operation *inst,
                                          Value instValue,
                                          SmallVectorImpl<Value> &newOperands,
                                          ArrayRef<Backedge> newResults) {
  // Create a ChannelStructsToPortOp to wrap the individual channel signals
  // that the output axi4.port has now been split into back into an axi4.port
  // outside the module
  auto [clock, reset] = filler.get(inst->getBlock());
  SmallVector<Value> operands{clock, reset};
  llvm::append_range(operands, instanceDriven(newResults));
  SmallVector<Type> resultTypes{portType};
  llvm::append_range(resultTypes, portTypes(inputPorts));

  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  auto toPort = ChannelStructsToPortOp::create(builder, resultTypes, operands);

  instValue.replaceAllUsesWith(toPort.getPort());
  for (auto [port, value] :
       llvm::zip_equal(inputPorts, toPort.getResults().drop_front()))
    newOperands[port.argNum] = value;
}

SmallVector<Type> AXI4StructPortConversion::portAndResponseTypes() {
  SmallVector<Type> types{portType};
  llvm::append_range(types, signalTypes(portType, /*managerDriven=*/false));
  return types;
}

/// Build the request and response ports for an !axi4.port input
void AXI4StructPortConversion::buildInputSignals() {
  // This hook is called when an !axi.port is an input to a module, so we're in
  // a subordinate: the manager drives the request and we drive the response
  Value request = converter.createNewInput(
      origPort, "_req", structType(portType, kRequestFields), reqPort);
  auto driveResponse = [&](Value value) {
    converter.createNewOutput(origPort, "_resp",
                              structType(portType, kResponseFields), value,
                              respPort);
  };

  // A module with no body (e.g. extern) can't do anything with values, so just
  // add the ports
  if (!body)
    return driveResponse({});

  // Unpack the request into the signals the !axi.port value that was already
  // being digested is built from
  auto [clock, reset] = filler.get(body);
  ImplicitLocOpBuilder b(origPort.loc, body->getTerminator());
  SmallVector<Value> operands{clock, reset};
  llvm::append_range(operands,
                     explodeStruct(b, portType, kRequestFields, request));

  auto toPort =
      ChannelStructsToPortOp::create(b, portAndResponseTypes(), operands);
  body->getArgument(origPort.argNum).replaceAllUsesWith(toPort.getPort());
  driveResponse(packStruct(b, portType, kResponseFields,
                           toPort.getResults().drop_front()));
}

/// Build the request and response ports for an !axi4.port output
void AXI4StructPortConversion::buildOutputSignals() {
  // This hook is called when an !axi.port is an output of a module, so we're in
  // a manager: we drive the request and the subordinate drives the response
  Value response = converter.createNewInput(
      origPort, "_resp", structType(portType, kResponseFields), respPort);
  auto driveRequest = [&](Value value) {
    converter.createNewOutput(
        origPort, "_req", structType(portType, kRequestFields), value, reqPort);
  };

  // A module with no body (e.g. extern) can't do anything with values, so just
  // add the ports
  if (!body)
    return driveRequest({});

  // Break the !axi.port value that was already being driven into the signals
  // the request is built from
  auto [clock, reset] = filler.get(body);
  Operation *terminator = body->getTerminator();
  ImplicitLocOpBuilder b(origPort.loc, terminator);
  SmallVector<Value> operands{clock, reset,
                              terminator->getOperand(origPort.argNum)};
  llvm::append_range(operands,
                     explodeStruct(b, portType, kResponseFields, response));

  auto fromPort = PortToChannelStructsOp::create(
      b, signalTypes(portType, /*managerDriven=*/true), operands,
      concurrency(b, portType));
  driveRequest(packStruct(b, portType, kRequestFields, fromPort.getResults()));
}

// Map to the newly created ports on the instances of the modified subordinate
void AXI4StructPortConversion::mapInputSignals(
    OpBuilder &b, Operation *inst, Value instValue,
    SmallVectorImpl<Value> &newOperands, ArrayRef<Backedge> newResults) {
  // Break down the port that was previously an input to the instance, and feed
  // the instance the request its signals make up
  auto [clock, reset] = filler.get(inst->getBlock());
  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  SmallVector<Value> operands{clock, reset, instValue};
  llvm::append_range(operands, explodeStruct(builder, portType, kResponseFields,
                                             newResults[respPort.argNum]));

  auto fromPort = PortToChannelStructsOp::create(
      builder, signalTypes(portType, /*managerDriven=*/true), operands,
      concurrency(builder, portType));
  newOperands[reqPort.argNum] =
      packStruct(builder, portType, kRequestFields, fromPort.getResults());
}

// Map to the newly created ports on the instances of the modified manager
void AXI4StructPortConversion::mapOutputSignals(
    OpBuilder &b, Operation *inst, Value instValue,
    SmallVectorImpl<Value> &newOperands, ArrayRef<Backedge> newResults) {
  // Wrap the request the instance now drives back into the !axi4.port it drove
  // before, and hand it the response that port's signals make up
  auto [clock, reset] = filler.get(inst->getBlock());
  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  SmallVector<Value> operands{clock, reset};
  llvm::append_range(operands, explodeStruct(builder, portType, kRequestFields,
                                             newResults[reqPort.argNum]));

  auto toPort =
      ChannelStructsToPortOp::create(builder, portAndResponseTypes(), operands);
  instValue.replaceAllUsesWith(toPort.getPort());
  newOperands[respPort.argNum] = packStruct(builder, portType, kResponseFields,
                                            toPort.getResults().drop_front());
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Report a port this pass has no signals to wire to.
static LogicalResult checkLowerable(Value port) {
  if (!isa<PortType>(port.getType()) || !port.use_empty())
    return success();
  return mlir::emitError(port.getLoc())
         << "AXI4 port has no uses, so cannot be lowered";
}

/// Warn about each bodyless module we are about to change the ports of, since
/// we cannot directly update its implementation at the same time to ensure they
/// match
static void warnPortsChanging(ModuleOp module) {
  for (auto mod : module.getOps<hw::HWMutableModuleLike>()) {
    if (mod.getBodyBlock())
      continue;
    SmallVector<StringRef> names;
    for (const hw::PortInfo &port : mod.getPortList())
      if (isa<PortType>(port.type))
        names.push_back(port.getName());
    if (names.empty())
      continue;

    auto diag = mod->emitWarning()
                << "lowering AXI4 port" << (names.size() == 1 ? " " : "s ");
    llvm::interleaveComma(names, diag,
                          [&](StringRef name) { diag << "'" << name << "'"; });
    diag << " changes the ports of this module; its implementation must match "
            "the new port list";
  }
}

namespace {
struct AXI4ToHWPass : public circt::impl::AXI4ToHWBase<AXI4ToHWPass> {
  using AXI4ToHWBase::AXI4ToHWBase;

  void runOnOperation() override;
};
} // namespace

void AXI4ToHWPass::runOnOperation() {
  ModuleOp module = getOperation();
  bool anyFailed = false;

  // Pre-walk to catch cases that we can't lower
  module.walk([&](Operation *op) {
    if (isa<AbstractManagerOp, AbstractSubordinateOp>(op)) {
      op->emitOpError("models an endpoint with no RTL, so cannot be lowered");
      anyFailed = true;
    }

    for (Value result : op->getResults())
      anyFailed |= failed(checkLowerable(result));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          anyFailed |= failed(checkLowerable(arg));
  });
  if (anyFailed)
    return signalPassFailure();

  warnPortsChanging(module);

  // Components become instances, so they have to land before the instance graph
  // analysis is generated
  DenseSet<Operation *> componentModules;
  if (failed(lowerComponents(module, pulpMapping, componentModules)))
    return signalPassFailure();

  FillerDomain filler;
  hw::InstanceGraph &instanceGraph = getAnalysis<hw::InstanceGraph>();
  for (auto mod : module.getOps<hw::HWMutableModuleLike>()) {
    // The component externs are internal to the lowering, so they keep their
    // signals however the boundary is expressed.
    bool structPorts =
        reqRespPorts && !componentModules.contains(mod.getOperation());
    if (failed(hw::PortConverter<AXI4PortConversionBuilder>(instanceGraph, mod,
                                                            filler, structPorts)
                   .run()))
      return signalPassFailure();
  }

  if (failed(annihilateBridges(module, filler)))
    return signalPassFailure();
  filler.eraseUnused();

  // Ensure we successfully lowered all axi4 ops
  module.walk([&](Operation *op) {
    if (isa_and_nonnull<AXI4Dialect>(op->getDialect())) {
      op->emitOpError("could not be lowered to HW");
      anyFailed = true;
    }
  });
  if (anyFailed)
    signalPassFailure();
}
