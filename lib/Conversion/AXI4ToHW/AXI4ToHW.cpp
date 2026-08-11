//===- AXI4ToHW.cpp - AXI4 to HW conversion pass --------------------------===//
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

/// The types of `ports`, in order.
static SmallVector<Type> portTypes(ArrayRef<hw::PortInfo> ports) {
  return llvm::map_to_vector(
      ports, [](const hw::PortInfo &port) { return port.type; });
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
  /// The clock and reset ports it takes, in order, and what drives them
  SmallVector<std::pair<StringRef, Value>> domain;
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
      .Default(std::nullopt);
}

/// The ports of the module implementing `component`. A component is the
/// subordinate to its upstream managers, so upstream ports are inputs and
/// downstream ports are outputs.
static SmallVector<hw::ModulePort> componentPorts(const Component &component) {
  MLIRContext *context = component.op->getContext();
  SmallVector<hw::ModulePort> ports;
  for (auto [name, value] : component.domain)
    ports.push_back({StringAttr::get(context, name), value.getType(),
                     hw::ModulePort::Direction::Input});
  for (auto [index, value] : llvm::enumerate(upstreamPorts(component.op)))
    ports.push_back({StringAttr::get(context, "mgr" + Twine(index)),
                     value.getType(), hw::ModulePort::Direction::Input});
  for (auto [index, value] : llvm::enumerate(component.op->getResults()))
    ports.push_back({StringAttr::get(context, "sub" + Twine(index)),
                     value.getType(), hw::ModulePort::Direction::Output});
  return ports;
}

/// Replace every component with an instance of an external module of its shape,
/// shared by components of the same kind whose ports match.
static LogicalResult lowerComponents(ModuleOp module, bool pulpMapping) {
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
      if (pulpMapping)
        attachPulpSource(b, shape, op);
    }

    SmallVector<Value> inputs = llvm::map_to_vector(
        component.domain, [](const std::pair<StringRef, Value> &domain) {
          return domain.second;
        });
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
/// Placeholder clock and reset for struct<->port ops that are temporarily
/// materialized between PortConverter invocations
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

/// Wire through and erase every back-to-back bridge pair - errors if there's a
/// domain crossing.
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

    // Complain if we have two non-filler converter back to back in different
    // domains
    if (!filler.isFiller(toPort.getClock()) &&
        !filler.isFiller(fromPort.getClock())) {
      if (toPort.getClock() != fromPort.getClock())
        return emitDomainCrossing(fromPort, toPort, "clock");
      if (toPort.getReset() != fromPort.getReset())
        return emitDomainCrossing(fromPort, toPort, "reset");
    }

    // Each op returns the signals the other was given: the manager-driven ones
    // one way and the subordinate-driven ones the other. Both operand lists
    // drop the clock and the reset, and `fromPort` also drops its port operand
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
  /// The types of the exploded signals this module drives.
  SmallVector<Type> drivenTypes();
  /// The signals the new instance drives for this port, in signal order. They
  /// come back as backedges because the instance is not built yet.
  SmallVector<Value> instanceDriven(ArrayRef<Backedge> newResults);

  FillerDomain &filler;
  PortType portType;
  /// The generated ports, split by this module's own direction, in AXI signal
  /// order
  SmallVector<hw::PortInfo> inputPorts, outputPorts;
};

class AXI4PortConversionBuilder : public hw::PortConversionBuilder {
public:
  AXI4PortConversionBuilder(hw::PortConverterImpl &converter,
                            FillerDomain &filler)
      : PortConversionBuilder(converter), filler(filler) {}

  FailureOr<std::unique_ptr<hw::PortConversion>>
  build(hw::PortInfo port) override {
    if (isa<PortType>(port.type))
      return {std::make_unique<AXI4PortConversion>(converter, port, filler)};
    return PortConversionBuilder::build(port);
  }

private:
  FillerDomain &filler;
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

SmallVector<Type> AXI4PortConversion::drivenTypes() {
  SmallVector<Type> types;
  for (const SignalInfo &signal : portSignals(portType))
    if (signal.managerDrives == isManager())
      types.push_back(signal.type);
  return types;
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
  llvm::append_range(resultTypes, drivenTypes());

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
  auto fromPort = PortToChannelStructsOp::create(b, drivenTypes(), operands);
  createOutputPorts(fromPort.getResults());
}

// Map to the newly created ports on the instances of the modified subordinate
void AXI4PortConversion::mapInputSignals(OpBuilder &b, Operation *inst,
                                         Value instValue,
                                         SmallVectorImpl<Value> &newOperands,
                                         ArrayRef<Backedge> newResults) {
  // The instance takes a port: convert it into the signals the new instance
  // takes, driven by the ones it produces.
  auto [clock, reset] = filler.get(inst->getBlock());
  SmallVector<Value> operands{clock, reset, instValue};
  llvm::append_range(operands, instanceDriven(newResults));

  ImplicitLocOpBuilder builder(origPort.loc, b.getInsertionBlock(),
                               b.getInsertionPoint());
  auto fromPort =
      PortToChannelStructsOp::create(builder, portTypes(inputPorts), operands);

  for (auto [port, value] : llvm::zip_equal(inputPorts, fromPort.getResults()))
    newOperands[port.argNum] = value;
}

// Map to the newly created ports on the instances of the modified manager
void AXI4PortConversion::mapOutputSignals(OpBuilder &b, Operation *inst,
                                          Value instValue,
                                          SmallVectorImpl<Value> &newOperands,
                                          ArrayRef<Backedge> newResults) {
  // The instance produces a port: convert the signals the new instance
  // produces into one, and feed it back the ones it takes.
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

/// Warn about each bodyless module we are about to change the ports of - its
/// implementation is not ours to rewrite, so changing them silently would
/// leave it mismatched against whatever supplies it.
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

  // Reject what cannot be lowered before mutating anything.
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
  if (failed(lowerComponents(module, pulpMapping)))
    return signalPassFailure();

  FillerDomain filler;
  hw::InstanceGraph &instanceGraph = getAnalysis<hw::InstanceGraph>();
  for (auto mod : module.getOps<hw::HWMutableModuleLike>())
    if (failed(hw::PortConverter<AXI4PortConversionBuilder>(instanceGraph, mod,
                                                            filler)
                   .run()))
      return signalPassFailure();

  if (failed(annihilateBridges(module, filler)))
    return signalPassFailure();
  filler.eraseUnused();

  // A bridge that failed to pair up would carry a filler clock into the output.
  module.walk([&](Operation *op) {
    if (isa_and_nonnull<AXI4Dialect>(op->getDialect())) {
      op->emitOpError("could not be lowered to HW");
      anyFailed = true;
    }
  });
  if (anyFailed)
    signalPassFailure();
}
