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
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
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

/// One AXI interface to wire on an instance: its role, port type, port_wires
/// name, and clock. `diag` is the op errors are reported against.
struct PortGroupSpec {
  Operation *diag;
  bool isManager;
  PortType portType;
  std::string name;
  Value axiClock;
  std::string clockPort;
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
// PULP axi_xbar interface modeling
//
// The abstract xbar lowers to an instance of PULP's `axi_xbar`, whose slave and
// master faces each carry an array of packed req/resp structs. Everything below
// models that interface and bridges it to our canonical channel form.
//===----------------------------------------------------------------------===//

// PULP's aw_chan_t carries an atomic-op field our canonical AW payload lacks;
// every channel carries a `user` sideband. Both are tied to 0 on lowering. The
// `user` width is a stopgap: `!axi4.port` carries no user width, so we hardcode
// it to 1.
constexpr unsigned kAtopWidth = 6;
constexpr unsigned kUserWidth = 1;

/// The PULP channel struct type for one channel: the canonical payload plus a
/// `user` field on every channel and a 6-bit `atop` on AW, in `AXI_TYPEDEF_*`
/// field order (atop before user).
hw::StructType getPulpChannelType(PortType port, AXI4Channel channel) {
  hw::StructType canon = getChannelPayloadType(port, channel);
  MLIRContext *ctx = port.getContext();
  SmallVector<hw::StructType::FieldInfo> fields(canon.getElements());
  if (channel == AXI4Channel::AW)
    fields.push_back({StringAttr::get(ctx, "atop"),
                      IntegerType::get(ctx, kAtopWidth)});
  fields.push_back(
      {StringAttr::get(ctx, "user"), IntegerType::get(ctx, kUserWidth)});
  return hw::StructType::get(ctx, fields);
}

/// PULP's `req_t`: all manager->subordinate signals for one port, in the field
/// order `AXI_TYPEDEF_REQ_T` emits.
hw::StructType getPulpReqType(PortType port) {
  MLIRContext *ctx = port.getContext();
  Type i1 = IntegerType::get(ctx, 1);
  auto f = [&](StringRef n, Type t) {
    return hw::StructType::FieldInfo{StringAttr::get(ctx, n), t};
  };
  return hw::StructType::get(
      ctx, {f("aw", getPulpChannelType(port, AXI4Channel::AW)), f("aw_valid", i1),
            f("w", getPulpChannelType(port, AXI4Channel::W)), f("w_valid", i1),
            f("b_ready", i1),
            f("ar", getPulpChannelType(port, AXI4Channel::AR)), f("ar_valid", i1),
            f("r_ready", i1)});
}

/// PULP's `resp_t`: all subordinate->manager signals for one port, in the field
/// order `AXI_TYPEDEF_RESP_T` emits.
hw::StructType getPulpRespType(PortType port) {
  MLIRContext *ctx = port.getContext();
  Type i1 = IntegerType::get(ctx, 1);
  auto f = [&](StringRef n, Type t) {
    return hw::StructType::FieldInfo{StringAttr::get(ctx, n), t};
  };
  return hw::StructType::get(
      ctx, {f("aw_ready", i1), f("ar_ready", i1), f("w_ready", i1),
            f("b_valid", i1), f("b", getPulpChannelType(port, AXI4Channel::B)),
            f("r_valid", i1), f("r", getPulpChannelType(port, AXI4Channel::R))});
}

/// Wrap a canonical channel payload as PULP's channel struct, tying the extra
/// `user` (and `atop` on AW) fields to 0.
Value toPulpChannel(ImplicitLocOpBuilder &b, AXI4Channel channel,
                    Value canonical) {
  auto canonTy = cast<hw::StructType>(canonical.getType());
  SmallVector<Value> fields;
  for (auto &fi : canonTy.getElements())
    fields.push_back(hw::StructExtractOp::create(b, canonical, fi.name));
  SmallVector<hw::StructType::FieldInfo> pulpFields(canonTy.getElements());
  if (channel == AXI4Channel::AW) {
    fields.push_back(hw::ConstantOp::create(b, b.getIntegerType(kAtopWidth), 0));
    pulpFields.push_back({b.getStringAttr("atop"), b.getIntegerType(kAtopWidth)});
  }
  fields.push_back(hw::ConstantOp::create(b, b.getIntegerType(kUserWidth), 0));
  pulpFields.push_back({b.getStringAttr("user"), b.getIntegerType(kUserWidth)});
  return hw::StructCreateOp::create(
      b, hw::StructType::get(b.getContext(), pulpFields), fields);
}

/// Drop the PULP-only fields (`user`, and `atop` on AW) off a PULP channel
/// struct, yielding a canonical channel payload. The canonical fields are a
/// by-name subset of the PULP struct, so one extraction loop covers every
/// channel.
Value fromPulpChannel(ImplicitLocOpBuilder &b, Value pulp,
                      hw::StructType canonTy) {
  SmallVector<Value> fields;
  for (auto &fi : canonTy.getElements())
    fields.push_back(hw::StructExtractOp::create(b, pulp, fi.name));
  return hw::StructCreateOp::create(b, canonTy, fields);
}

/// Build one PULP xbar face's input-array struct element (`req_t` at a
/// subordinate face, `resp_t` at a manager face) from fresh backedges,
/// populating the received signals of `wires`. The driven signals are left null
/// and filled by `fillXbarFaceOutputs` once the instance exists.
Value buildXbarFaceInput(bool isManager, PortType portType,
                         ImplicitLocOpBuilder &builder, BackedgeBuilder &bb,
                         PortWires &wires) {
  Type i1 = builder.getI1Type();
  hw::StructType inputTy =
      isManager ? getPulpRespType(portType) : getPulpReqType(portType);

  DenseMap<StringAttr, Value> fields;
  wires.resize(kNumChannels);
  for (auto [idx, ci] : llvm::enumerate(kChannelInfos)) {
    ChannelWires &cw = wires[idx];
    bool payloadDriven = (isManager == ci.isRequest);
    if (payloadDriven) {
      // payload/valid are driven by the xbar (filled later); ready is received.
      Backedge ready = bb.get(i1);
      cw.ready = {ready, ready};
      fields[builder.getStringAttr(ci.token + Twine("_ready"))] = ready;
      continue;
    }
    // payload/valid are received; ready is driven by the xbar (filled later).
    Backedge payload = bb.get(getChannelPayloadType(portType, ci.channel));
    cw.payload = {payload, payload};
    Backedge valid = bb.get(i1);
    cw.valid = {valid, valid};
    Value payloadField = toPulpChannel(builder, ci.channel, payload);
    fields[builder.getStringAttr(ci.token)] = payloadField;
    fields[builder.getStringAttr(ci.token + Twine("_valid"))] = valid;
  }

  SmallVector<Value> ordered;
  for (auto &fi : inputTy.getElements())
    ordered.push_back(fields.at(fi.name));
  return hw::StructCreateOp::create(builder, inputTy, ordered);
}

/// Fill the driven signals of `wires` from one xbar face's output-array struct
/// element (`resp_t` at a subordinate face, `req_t` at a manager face).
void fillXbarFaceOutputs(bool isManager, PortType portType,
                         ImplicitLocOpBuilder &builder, Value outputStruct,
                         PortWires &wires) {
  for (auto [idx, ci] : llvm::enumerate(kChannelInfos)) {
    ChannelWires &cw = wires[idx];
    bool payloadDriven = (isManager == ci.isRequest);
    std::string token = ci.token.str();
    if (payloadDriven) {
      Value payload = hw::StructExtractOp::create(builder, outputStruct, token);
      payload = fromPulpChannel(builder, payload,
                                getChannelPayloadType(portType, ci.channel));
      cw.payload.value = payload;
      cw.valid.value =
          hw::StructExtractOp::create(builder, outputStruct, token + "_valid");
      continue;
    }
    cw.ready.value =
        hw::StructExtractOp::create(builder, outputStruct, token + "_ready");
  }
}

//===----------------------------------------------------------------------===//
// PULP axi_xbar wrapper generation
//
// PULP's `axi_xbar` can't be instantiated straight from `hw`: it is
// parameterized by struct-typed `Cfg`/`rule_t` values and a compile-time address
// map that `hw` parameters can't express. Instead we emit a generated
// SystemVerilog wrapper (an `sv.verbatim.source`) that declares the AXI typedefs,
// bakes in the `Cfg` and address map, ties off reset/test/default-port controls,
// and instantiates `axi_xbar`. A companion `sv.verbatim.module` gives it the
// typed data-plane interface that the `hw.instance` targets. All typedefs are
// prefixed with the wrapper name so several wrappers can share a compilation
// unit without $unit-scope collisions.
//===----------------------------------------------------------------------===//

/// One `xbar_rule_*_t` entry: requests in [start, end) route to master port
/// `idx`.
struct AddrRule {
  unsigned idx;
  uint64_t start;
  uint64_t end;
};

/// Render the SystemVerilog wrapper for one crossbar configuration.
std::string buildXbarWrapperSource(StringRef name, unsigned numUp,
                                   unsigned numDown, PortType upType,
                                   PortType downType, ArrayRef<AddrRule> rules) {
  unsigned addrW = upType.getAddressWidth();
  unsigned dataW = upType.getDataWidth();
  unsigned slvId = upType.getIdWidth();
  unsigned mstId = downType.getIdWidth();
  unsigned ruleW = addrW <= 32 ? 32 : 64;
  std::string ruleTy =
      (Twine("axi_pkg::xbar_rule_") + Twine(ruleW) + "_t").str();
  std::string p = (name + "_").str();
  auto hex = [&](uint64_t v) {
    return (Twine(ruleW) + "'h" + llvm::utohexstr(v)).str();
  };

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// Generated by --lower-axi4-to-hw: wrapper instantiating PULP "
        "axi_xbar.\n";
  os << "`include \"axi/typedef.svh\"\n\n";

  // AXI typedefs at file scope, name-prefixed to stay collision-free.
  os << "typedef logic [" << addrW << "-1:0] " << p << "addr_t;\n";
  os << "typedef logic [" << dataW << "-1:0] " << p << "data_t;\n";
  os << "typedef logic [" << dataW << "/8-1:0] " << p << "strb_t;\n";
  os << "typedef logic [" << kUserWidth << "-1:0] " << p << "user_t;\n";
  os << "typedef logic [" << slvId << "-1:0] " << p << "slv_id_t;\n";
  os << "typedef logic [" << mstId << "-1:0] " << p << "mst_id_t;\n";
  os << "`AXI_TYPEDEF_ALL(" << p << "slv, " << p << "addr_t, " << p
     << "slv_id_t, " << p << "data_t, " << p << "strb_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_ALL(" << p << "mst, " << p << "addr_t, " << p
     << "mst_id_t, " << p << "data_t, " << p << "strb_t, " << p << "user_t)\n\n";

  // The data-plane interface mirrors the `sv.verbatim.module` port list.
  os << "module " << name << " (\n";
  os << "  input  logic clk_i,\n";
  os << "  input  " << p << "slv_req_t  [" << numUp << "-1:0] slv_ports_req_i,\n";
  os << "  output " << p << "slv_resp_t [" << numUp
     << "-1:0] slv_ports_resp_o,\n";
  os << "  output " << p << "mst_req_t  [" << numDown
     << "-1:0] mst_ports_req_o,\n";
  os << "  input  " << p << "mst_resp_t [" << numDown
     << "-1:0] mst_ports_resp_i\n";
  os << ");\n";

  // Baked-in configuration. `default: '0` covers version-specific Cfg fields
  // (e.g. PipelineStages) we don't set.
  os << "  localparam axi_pkg::xbar_cfg_t Cfg = '{\n";
  os << "    NoSlvPorts:         " << numUp << ",\n";
  os << "    NoMstPorts:         " << numDown << ",\n";
  os << "    MaxMstTrans:        8,\n";
  os << "    MaxSlvTrans:        8,\n";
  os << "    FallThrough:        1'b0,\n";
  os << "    LatencyMode:        axi_pkg::CUT_ALL_AX,\n";
  os << "    AxiIdWidthSlvPorts: " << slvId << ",\n";
  os << "    AxiIdUsedSlvPorts:  " << slvId << ",\n";
  os << "    UniqueIds:          1'b0,\n";
  os << "    AxiAddrWidth:       " << addrW << ",\n";
  os << "    AxiDataWidth:       " << dataW << ",\n";
  os << "    NoAddrRules:        " << rules.size() << ",\n";
  os << "    default:            '0\n";
  os << "  };\n";

  // Address map derived from the downstream endpoints' access windows.
  os << "  localparam " << ruleTy << " [" << rules.size()
     << "-1:0] AddrMap = '{\n";
  for (auto [i, r] : llvm::enumerate(rules))
    os << "    '{idx: " << r.idx << ", start_addr: " << hex(r.start)
       << ", end_addr: " << hex(r.end) << "}"
       << (i + 1 < rules.size() ? "," : "") << "\n";
  os << "  };\n";

  // Instantiate the real crossbar; reset/test/default-port controls tied off.
  os << "  axi_xbar #(\n";
  os << "    .Cfg           (Cfg),\n";
  os << "    .ATOPs         (1'b1),\n";
  os << "    .Connectivity  ('1),\n";
  os << "    .slv_aw_chan_t (" << p << "slv_aw_chan_t),\n";
  os << "    .mst_aw_chan_t (" << p << "mst_aw_chan_t),\n";
  os << "    .w_chan_t      (" << p << "slv_w_chan_t),\n";
  os << "    .slv_b_chan_t  (" << p << "slv_b_chan_t),\n";
  os << "    .mst_b_chan_t  (" << p << "mst_b_chan_t),\n";
  os << "    .slv_ar_chan_t (" << p << "slv_ar_chan_t),\n";
  os << "    .mst_ar_chan_t (" << p << "mst_ar_chan_t),\n";
  os << "    .slv_r_chan_t  (" << p << "slv_r_chan_t),\n";
  os << "    .mst_r_chan_t  (" << p << "mst_r_chan_t),\n";
  os << "    .slv_req_t     (" << p << "slv_req_t),\n";
  os << "    .slv_resp_t    (" << p << "slv_resp_t),\n";
  os << "    .mst_req_t     (" << p << "mst_req_t),\n";
  os << "    .mst_resp_t    (" << p << "mst_resp_t),\n";
  os << "    .rule_t        (" << ruleTy << ")\n";
  os << "  ) i_xbar (\n";
  os << "    .clk_i                 (clk_i),\n";
  os << "    .rst_ni                (1'b1),\n";
  os << "    .test_i                (1'b0),\n";
  os << "    .slv_ports_req_i       (slv_ports_req_i),\n";
  os << "    .slv_ports_resp_o      (slv_ports_resp_o),\n";
  os << "    .mst_ports_req_o       (mst_ports_req_o),\n";
  os << "    .mst_ports_resp_i      (mst_ports_resp_i),\n";
  os << "    .addr_map_i            (AddrMap),\n";
  os << "    .en_default_mst_port_i ('0),\n";
  os << "    .default_mst_port_i    ('0)\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
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
      Operation *portOp, bool isManager, StringRef name,
      DenseMap<StringRef, PortInfo> &portByName,
      DenseMap<StringRef, Value> &inputByName,
      SmallVectorImpl<std::pair<Wire *, unsigned>> &outs,
      SmallVectorImpl<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
          &outStructs,
      ImplicitLocOpBuilder &builder)
      : portOp(portOp),
        base(((isManager ? "m_axi_" : "s_axi_") + name + "_").str()),
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
  LogicalResult lowerXbar(XbarOp xbar);
  /// Get (or create) the PULP `axi_xbar` wrapper for this crossbar shape and
  /// address map, deduplicated by their combined signature.
  sv::SVVerbatimModuleOp getOrCreateXbarModule(unsigned numUpstream,
                                               unsigned numDownstream,
                                               PortType upstreamType,
                                               PortType downstreamType,
                                               ArrayRef<AddrRule> rules);
  /// Instantiate `moduleOp`, wiring each interface in `specs` via the flat
  /// port_wires convention; returns the per-interface wires in `wiresOut`.
  LogicalResult buildInstance(Operation *diag, hw::HWModuleLike moduleOp,
                              StringRef instanceName,
                              ArrayRef<PortGroupSpec> specs,
                              SmallVectorImpl<PortWires> &wiresOut);
  /// Materialize an `!axi4.clock` value as the module clock port's type.
  Value materializeClock(Value axiClock, Type portType);

  ModuleOp module;
  ImplicitLocOpBuilder builder;
  BackedgeBuilder bb;
  DenseMap<OpOperand *, EdgeWires> edges;
  DenseMap<Value, Value> clockCache;
  /// Emitted xbar wrappers, keyed by shape+address-map signature.
  llvm::StringMap<sv::SVVerbatimModuleOp> xbarWrappers;
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

LogicalResult
NetworkLowering::buildInstance(Operation *diag, hw::HWModuleLike moduleOp,
                              StringRef instanceName,
                              ArrayRef<PortGroupSpec> specs,
                              SmallVectorImpl<PortWires> &wiresOut) {
  DenseMap<StringRef, PortInfo> portByName;
  for (auto &p : moduleOp.getPortList())
    portByName[p.getName()] = p;

  DenseMap<StringRef, Value> inputByName;
  SmallVector<std::pair<Wire *, unsigned>> outs;
  SmallVector<std::tuple<Wire *, hw::StructType, SmallVector<unsigned>>>
      outStructs;

  // Reserve so the deferred output pointers below stay valid: `outs`/`outStructs`
  // hold Wire pointers into these elements until we resolve them post-instance.
  wiresOut.reserve(specs.size());
  for (const PortGroupSpec &spec : specs) {
    wiresOut.emplace_back();
    PortWiresMappingLowerer lowerer(spec.diag, spec.isManager, spec.name,
                                    portByName, inputByName, outs, outStructs,
                                    builder);
    if (failed(populatePortGroup(spec.isManager, spec.portType, builder, bb,
                                 wiresOut.back(), lowerer)))
      return failure();

    auto clockPort = portByName.find(spec.clockPort);
    if (clockPort == portByName.end())
      return spec.diag->emitError("referenced module has no clock port '")
             << spec.clockPort << "'";
    Value clock = materializeClock(spec.axiClock, clockPort->second.type);
    if (!clock)
      return spec.diag->emitError("unsupported clock port type");
    inputByName[clockPort->second.name.getValue()] = clock;
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

  // Resolve deferred module outputs into real driving values.
  for (auto &[wire, argNum] : outs)
    wire->value = inst.getResult(argNum);
  for (auto &[wire, structTy, fieldArgs] : outStructs) {
    SmallVector<Value> fields;
    for (unsigned argNum : fieldArgs)
      fields.push_back(inst.getResult(argNum));
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

    auto portWires = dyn_cast<PortWiresAttr>(mapping);
    if (!portWires)
      return portOp->emitError(
          "only the 'port_wires' port_mapping is currently supported");
    specs.push_back({portOp, isManager, portType, portWires.getName().str(),
                     axiClock, mapping.getClockPort().str()});
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
      edges[&*mgr.getPort().use_begin()].producer = std::move(wires);
    else
      edges[&portOp->getOpOperand(0)].consumer = std::move(wires);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PULP axi_xbar lowering
//===----------------------------------------------------------------------===//

sv::SVVerbatimModuleOp
NetworkLowering::getOrCreateXbarModule(unsigned numUpstream,
                                      unsigned numDownstream,
                                      PortType upstreamType,
                                      PortType downstreamType,
                                      ArrayRef<AddrRule> rules) {
  MLIRContext *ctx = module.getContext();
  std::string shape =
      ("axi_xbar_" + Twine(numUpstream) + "u" + Twine(numDownstream) + "d_a" +
       Twine(upstreamType.getAddressWidth()) + "_d" +
       Twine(upstreamType.getDataWidth()) + "_i" +
       Twine(upstreamType.getIdWidth()) + "_o" +
       Twine(downstreamType.getIdWidth()))
          .str();

  // Two xbars share a wrapper only if they route identically: the address map is
  // baked into the wrapper text, so it is part of the dedup signature.
  std::string signature = shape;
  {
    llvm::raw_string_ostream sig(signature);
    for (auto &r : rules)
      sig << "|" << r.idx << ":" << r.start << ":" << r.end;
  }
  if (auto existing = xbarWrappers.lookup(signature))
    return existing;

  // Name after the shape; disambiguate when a distinct routing reuses one.
  std::string name = shape;
  for (unsigned n = 0; module.lookupSymbol(name); ++n)
    name = (shape + "_" + Twine(n)).str();

  // The data-plane interface: the slave (upstream) ports face the managers, the
  // master (downstream) ports the subordinates. Each side packs its N interfaces
  // into an array of req/resp structs.
  auto arrayOf = [](hw::StructType elem, unsigned n) {
    return hw::ArrayType::get(elem, n);
  };
  auto port = [&](StringRef n, Type t, ModulePort::Direction d) {
    return PortInfo{{StringAttr::get(ctx, n), t, d}};
  };
  hw::StructType upReq = getPulpReqType(upstreamType);
  hw::StructType upResp = getPulpRespType(upstreamType);
  hw::StructType downReq = getPulpReqType(downstreamType);
  hw::StructType downResp = getPulpRespType(downstreamType);
  SmallVector<PortInfo> ports = {
      port("clk_i", IntegerType::get(ctx, 1), ModulePort::Input),
      port("slv_ports_req_i", arrayOf(upReq, numUpstream), ModulePort::Input),
      port("slv_ports_resp_o", arrayOf(upResp, numUpstream), ModulePort::Output),
      port("mst_ports_req_o", arrayOf(downReq, numDownstream),
           ModulePort::Output),
      port("mst_ports_resp_i", arrayOf(downResp, numDownstream),
           ModulePort::Input)};

  std::string source = buildXbarWrapperSource(name, numUpstream, numDownstream,
                                              upstreamType, downstreamType, rules);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto srcOp = sv::SVVerbatimSourceOp::create(
      builder, StringAttr::get(ctx, name + "_source"), source,
      hw::OutputFileAttr::getFromFilename(ctx, name + ".sv"),
      builder.getArrayAttr({}), /*additional_files=*/nullptr,
      builder.getStringAttr(name));
  auto modOp = sv::SVVerbatimModuleOp::create(
      builder, StringAttr::get(ctx, name), ports,
      FlatSymbolRefAttr::get(srcOp), builder.getArrayAttr({}),
      builder.getStringAttr(name));
  xbarWrappers[signature] = modOp;
  return modOp;
}

LogicalResult NetworkLowering::lowerXbar(XbarOp xbar) {
  builder.setLoc(xbar.getLoc());

  auto upstream = xbar.getUpstream();
  unsigned numUpstream = upstream.size();
  auto upstreamType = cast<PortType>(upstream.front().getType());
  auto downstreamType = cast<PortType>(xbar.getPort().getType());

  // A single result may fan out to several downstream consumers; each use is
  // one physical downstream port.
  SmallVector<OpOperand *> downstreamUses;
  for (OpOperand &use : xbar.getPort().getUses())
    downstreamUses.push_back(&use);
  unsigned numDownstream = downstreamUses.size();

  // Derive the address map from each downstream endpoint. A subordinate routes
  // its declared access windows; any other consumer (a chained xbar) has no
  // windows, so route the whole space to it (a documented stopgap).
  unsigned addrW = downstreamType.getAddressWidth();
  uint64_t fullRange = addrW >= 64 ? ~0ull : ((1ull << addrW) - 1);
  SmallVector<AddrRule> rules;
  for (unsigned j = 0; j < numDownstream; ++j) {
    if (auto sub = dyn_cast<SubordinatePortOp>(downstreamUses[j]->getOwner())) {
      for (auto win : sub.getAccess().getAsRange<WindowAttr>())
        rules.push_back({j, win.getBase(), win.getBase() + win.getSize()});
    } else {
      rules.push_back({j, 0, fullRange});
    }
  }

  auto moduleOp = getOrCreateXbarModule(numUpstream, numDownstream, upstreamType,
                                        downstreamType, rules);

  // Build each face's received-signal backedges and its input-array struct.
  // Upstream faces are subordinate-role; downstream faces are manager-role.
  SmallVector<PortWires, 0> upWires(numUpstream), downWires(numDownstream);
  SmallVector<Value> slvReq(numUpstream), mstResp(numDownstream);
  for (unsigned i = 0; i < numUpstream; ++i)
    slvReq[i] = buildXbarFaceInput(/*isManager=*/false, upstreamType, builder,
                                   bb, upWires[i]);
  for (unsigned j = 0; j < numDownstream; ++j)
    mstResp[j] = buildXbarFaceInput(/*isManager=*/true, downstreamType, builder,
                                    bb, downWires[j]);

  Value clock = materializeClock(xbar.getClock(), builder.getI1Type());
  assert(clock && "i1 clock always materializes");

  // Pack the per-face structs into the arrayed slave/master ports; array_create
  // takes elements in descending index order, so reverse.
  auto packArray = [&](ArrayRef<Value> elems) {
    SmallVector<Value> rev(llvm::reverse(elems));
    return hw::ArrayCreateOp::create(builder, rev).getResult();
  };
  DenseMap<StringRef, Value> inputByName;
  inputByName["clk_i"] = clock;
  inputByName["slv_ports_req_i"] = packArray(slvReq);
  inputByName["mst_ports_resp_i"] = packArray(mstResp);

  SmallVector<Value> inputs;
  for (auto &p : moduleOp.getPortList())
    if (p.isInput())
      inputs.push_back(inputByName.at(p.getName()));

  std::string instName = ("xbar_" + Twine(instanceCounter++)).str();
  auto inst = hw::InstanceOp::create(builder, moduleOp,
                                     builder.getStringAttr(instName), inputs);

  DenseMap<StringRef, Value> outputByName;
  unsigned resultIdx = 0;
  for (auto &p : moduleOp.getPortList())
    if (p.isOutput())
      outputByName[p.getName()] = inst.getResult(resultIdx++);

  // Unpack the driven signals for each face from the arrayed output ports.
  auto unpack = [&](Value array, unsigned idx, unsigned size) {
    unsigned width = std::max(1u, llvm::Log2_64_Ceil(size));
    Value index =
        hw::ConstantOp::create(builder, builder.getIntegerType(width), idx);
    return hw::ArrayGetOp::create(builder, array, index).getResult();
  };
  Value slvResp = outputByName.at("slv_ports_resp_o");
  Value mstReq = outputByName.at("mst_ports_req_o");
  for (unsigned i = 0; i < numUpstream; ++i)
    fillXbarFaceOutputs(/*isManager=*/false, upstreamType, builder,
                        unpack(slvResp, i, numUpstream), upWires[i]);
  for (unsigned j = 0; j < numDownstream; ++j)
    fillXbarFaceOutputs(/*isManager=*/true, downstreamType, builder,
                        unpack(mstReq, j, numDownstream), downWires[j]);

  // File the wires under their edges: an upstream interface is the consumer of
  // the operand feeding it; a downstream interface is the producer of a use.
  unsigned upstreamBase = upstream.getBeginOperandIndex();
  for (unsigned i = 0; i < numUpstream; ++i)
    edges[&xbar->getOpOperand(upstreamBase + i)].consumer =
        std::move(upWires[i]);
  for (unsigned j = 0; j < numDownstream; ++j)
    edges[downstreamUses[j]].producer = std::move(downWires[j]);
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
  SmallVector<NodeOp> nodes(module.getOps<NodeOp>());
  for (NodeOp node : nodes)
    if (failed(lowerNode(node)))
      return failure();

  SmallVector<XbarOp> xbars(module.getOps<XbarOp>());
  for (XbarOp xbar : xbars)
    if (failed(lowerXbar(xbar)))
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
