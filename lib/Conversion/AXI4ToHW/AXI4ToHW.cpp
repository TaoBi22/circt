//===- AXI4ToHW.cpp - Translate AXI4 into HW
//-------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Lowers abstract networks in the AXI4 dialect to HW
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
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
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

namespace {

// Define a canonical representation to store network signals in between
// components

/// One network signal that may or may not be a backedge.
struct Wire {
  Value value;
  std::optional<Backedge> backedge;
  bool isBackedge() const { return backedge.has_value(); }
};

enum class AXI4Channel { AW, W, B, AR, R };
constexpr unsigned kNumChannels = 5;

/// One AXI4 channel's three signals. `ready` flows opposite to
/// `payload`/`valid`.
struct ChannelWires {
  Wire payload;
  Wire valid;
  Wire ready;
};

/// A port's five channels, in canonical order (AW, W, B, AR, R).
using PortWires = std::array<ChannelWires, kNumChannels>;

/// The two sides of one AXI edge, keyed by the consuming operand. Each side is
/// set when its endpoint is lowered; both exist by the time we connect.
struct EdgeWires {
  std::optional<PortWires> producer;
  std::optional<PortWires> consumer;
};

/// How a port group's channels map onto the instantiated module's ports.
enum class MappingKind {
  /// Flat per-field scalar ports (port_wires attribute)
  PortWires,
  /// Struct-grouped channel-split ports (used as the interface for IP wrappers
  /// since it's close to the intermediate format we used, and sits in-between
  /// flat and everything being wrapped in one struct)
  ChannelPorts,
};

/// One AXI interface to wire on an instance
struct PortGroupSpec {
  // Defining op for diagnostics
  Operation *defOp;
  bool isManager;
  PortType portType;
  MappingKind kind;
  // Prefix for port names according to mapping - TODO: this will need building
  // out for req/resp ports
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

/// Per-channel lowering metadata. `token` is the port-name infix ("aw", "w",
/// etc); `isRequest` is true for channels a manager drives (AW, W, AR).
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

//===----------------------------------------------------------------------===//
// PULP lowering
//
// Lowers abstract AXI components to instantiations of PULP components (via an
// sv.verbatim wrapper to handle complex parameterizations that would be messy
// in hw)
//===----------------------------------------------------------------------===//

// TODO: when rebasing this on the up to date AXI dialect
constexpr unsigned kUserWidth = 1;

/// One `xbar_rule_*_t` entry: requests in [start, end) route to master port
/// `idx`. Following PULP's addr_decode, `end == 0` means "to the end of the
/// address space"
struct AddrRule {
  unsigned idx;
  uint64_t start;
  uint64_t end;
};

/// Get the exclusive end of the window - returns nullopt if window does not fit
/// space.
// TODO: assert addrW <= 64 once the AXI4 dialect verifies this (address
// widths wider than 64 bits can still reach this pass pre-rebase).
std::optional<uint64_t> windowEnd(uint64_t base, uint64_t size,
                                  unsigned addrW) {
  uint64_t end = base + size;
  bool overflow = end < base;
  if (addrW >= 64) {
    // Fits unless the sum passes 2^64
    if (!overflow)
      return end;
    return end == 0 ? std::optional<uint64_t>(0) : std::nullopt;
  }
  uint64_t top = 1ull << addrW;
  if (overflow || base >= top || end > top)
    return std::nullopt;
  return end == top ? 0 : end;
}

/// End of a rule for ordering/overlap math
uint64_t ruleEndValue(const AddrRule &r) { return r.end == 0 ? ~0ull : r.end; }

/// Append the [start, end) address ranges served by one downstream endpoint of
/// a crossbar.
void collectDownstreamRanges(
    Operation *consumer, unsigned addrW,
    SmallVectorImpl<std::pair<uint64_t, uint64_t>> &ranges,
    SmallPtrSetImpl<Operation *> &visited) {
  if (auto sub = dyn_cast<SubordinatePortOp>(consumer)) {
    for (auto win : sub.getAccess().getAsRange<WindowAttr>())
      ranges.push_back(
          {win.getBase(),
           windowEnd(win.getBase(), win.getSize(), addrW).value_or(0)});
    return;
  }
  // xbars serve the union of their downstream ranges
  if (auto xbar = dyn_cast<XbarOp>(consumer)) {
    if (!visited.insert(xbar).second)
      return; // Guard against pathological cycles.
    for (OpOperand &use : xbar.getPort().getUses())
      collectDownstreamRanges(use.getOwner(), addrW, ranges, visited);
  }
}

/// Derive the crossbar's address map: one rule per address range served by each
/// downstream endpoint
SmallVector<AddrRule> deriveXbarAddrMap(XbarOp xbar) {
  auto downType = cast<PortType>(xbar.getPort().getType());
  unsigned addrW = downType.getAddressWidth();
  SmallVector<AddrRule> rules;
  unsigned idx = 0;
  for (OpOperand &use : xbar.getPort().getUses()) {
    SmallVector<std::pair<uint64_t, uint64_t>> ranges;
    SmallPtrSet<Operation *, 4> visited;
    collectDownstreamRanges(use.getOwner(), addrW, ranges, visited);
    for (auto [start, end] : ranges)
      rules.push_back({idx, start, end});
    ++idx;
  }
  return rules;
}

/// SystemVerilog type name for one canonical channel-payload field, matching
/// the integer widths `getChannelPayloadType` uses.
std::string svChannelFieldType(StringRef field, StringRef prefix,
                               StringRef idTy) {
  if (field == "id")
    return idTy.str();
  // Use prefixed typedefs
  if (field == "addr")
    return (prefix + "addr_t").str();
  if (field == "data")
    return (prefix + "data_t").str();
  if (field == "strb")
    return (prefix + "strb_t").str();
  if (field == "len")
    return "axi_pkg::len_t";
  if (field == "size")
    return "axi_pkg::size_t";
  if (field == "burst")
    return "axi_pkg::burst_t";
  if (field == "cache")
    return "axi_pkg::cache_t";
  if (field == "prot")
    return "axi_pkg::prot_t";
  if (field == "qos")
    return "axi_pkg::qos_t";
  if (field == "region")
    return "axi_pkg::region_t";
  if (field == "resp")
    return "axi_pkg::resp_t";
  // lock, last
  return "logic";
}

/// Produce the SystemVerilog wrapper for one crossbar configuration.
std::string buildXbarWrapperSource(StringRef name, unsigned numUp,
                                   unsigned numDown, PortType upType,
                                   PortType downType,
                                   ArrayRef<AddrRule> rules) {
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
  // PULP-shaped req/resp/channel typedefs, used only internally to instantiate
  // axi_xbar.
  os << "`AXI_TYPEDEF_ALL(" << p << "slv, " << p << "addr_t, " << p
     << "slv_id_t, " << p << "data_t, " << p << "strb_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_ALL(" << p << "mst, " << p << "addr_t, " << p
     << "mst_id_t, " << p << "data_t, " << p << "strb_t, " << p << "user_t)\n";

  // Typedefs for the channel structs we use as wrapper ports
  auto emitCanonTypedef = [&](StringRef role, StringRef idTy, PortType pt,
                              const ChannelInfo &ci) {
    os << "typedef struct packed {";
    for (auto &fi : getChannelPayloadType(pt, ci.channel).getElements())
      os << " " << svChannelFieldType(fi.name.getValue(), p, idTy) << " "
         << fi.name.getValue() << ";";
    os << " } " << p << role << "_" << ci.token << "_t;\n";
  };
  std::string subId = (p + "slv_id_t");
  std::string mstIdTy = (p + "mst_id_t");
  for (const ChannelInfo &ci : kChannelInfos)
    emitCanonTypedef("sub", subId, upType, ci);
  for (const ChannelInfo &ci : kChannelInfos)
    emitCanonTypedef("mgr", mstIdTy, downType, ci);
  os << "\n";

  // Mirror the representation of channels payloads that we use between
  // components (matching buildChannelPortList).
  SmallVector<std::string> portDecls;
  portDecls.push_back("  input  logic clk_i");
  auto emitFacePorts = [&](StringRef role, unsigned idx, PortType pt,
                           bool isManager) {
    for (const ChannelInfo &ci : kChannelInfos) {
      bool payloadDrivenByModule = (isManager == ci.isRequest);
      StringRef fwd = payloadDrivenByModule ? "output" : "input ";
      StringRef rev = payloadDrivenByModule ? "input " : "output";
      std::string base = (role + Twine(idx) + "_" + ci.token).str();
      std::string ty = (p + role + "_" + ci.token + "_t").str();
      portDecls.push_back(("  " + fwd + " " + ty + " " + base).str());
      portDecls.push_back(("  " + fwd + " logic " + base + "_valid").str());
      portDecls.push_back(("  " + rev + " logic " + base + "_ready").str());
    }
  };
  for (unsigned i = 0; i < numUp; ++i)
    emitFacePorts("sub", i, upType, /*isManager=*/false);
  for (unsigned j = 0; j < numDown; ++j)
    emitFacePorts("mgr", j, downType, /*isManager=*/true);
  os << "module " << name << " (\n";
  os << llvm::join(portDecls, ",\n") << "\n";
  os << ");\n";

  // Internal PULP-shaped req/resp arrays: the boundary ports bridge to these,
  // and axi_xbar is wired directly to them.
  os << "  " << p << "slv_req_t  [" << numUp << "-1:0] slv_req;\n";
  os << "  " << p << "slv_resp_t [" << numUp << "-1:0] slv_resp;\n";
  os << "  " << p << "mst_req_t  [" << numDown << "-1:0] mst_req;\n";
  os << "  " << p << "mst_resp_t [" << numDown << "-1:0] mst_resp;\n";

  // Bridge each face's canonical channel ports to the PULP req/resp structs,
  // tying PULP's atop/user to 0 on the way in and dropping them on the way out.
  auto payloadPattern = [&](StringRef lhsExpr, StringRef rhsExpr, PortType pt,
                            const ChannelInfo &ci, bool toPulp) -> std::string {
    std::string s = "'{";
    bool first = true;
    for (auto &fi : getChannelPayloadType(pt, ci.channel).getElements()) {
      if (!first)
        s += ", ";
      first = false;
      s += (fi.name.getValue() + ": " + rhsExpr + "." + fi.name.getValue())
               .str();
    }
    if (toPulp) {
      if (ci.channel == AXI4Channel::AW)
        s += ", atop: '0";
      s += ", user: '0";
    }
    s += "}";
    return ("  assign " + lhsExpr + " = " + s + ";\n").str();
  };
  auto emitFaceBridge = [&](StringRef role, unsigned idx, PortType pt,
                            bool isManager) {
    std::string reqVar =
        ((isManager ? "mst_req[" : "slv_req[") + Twine(idx) + "]").str();
    std::string respVar =
        ((isManager ? "mst_resp[" : "slv_resp[") + Twine(idx) + "]").str();
    for (const ChannelInfo &ci : kChannelInfos) {
      bool payloadDrivenByModule = (isManager == ci.isRequest);
      std::string tok = ci.token.str();
      std::string port = (role + Twine(idx) + "_" + tok).str();
      // payload+valid live in the req struct for request channels, the resp
      // struct for response channels; ready lives in the other one.
      std::string pvVar = ci.isRequest ? reqVar : respVar;
      std::string rdyVar = ci.isRequest ? respVar : reqVar;
      if (payloadDrivenByModule) {
        // Wrapper drives the port from PULP (drop atop/user).
        os << payloadPattern(port, pvVar + "." + tok, pt, ci, /*toPulp=*/false);
        os << "  assign " << port << "_valid = " << pvVar << "." << tok
           << "_valid;\n";
        os << "  assign " << rdyVar << "." << tok << "_ready = " << port
           << "_ready;\n";
      } else {
        // Wrapper receives the port into PULP (tie atop/user to 0).
        os << payloadPattern(pvVar + "." + tok, port, pt, ci, /*toPulp=*/true);
        os << "  assign " << pvVar << "." << tok << "_valid = " << port
           << "_valid;\n";
        os << "  assign " << port << "_ready = " << rdyVar << "." << tok
           << "_ready;\n";
      }
    }
  };
  for (unsigned i = 0; i < numUp; ++i)
    emitFaceBridge("sub", i, upType, /*isManager=*/false);
  for (unsigned j = 0; j < numDown; ++j)
    emitFaceBridge("mgr", j, downType, /*isManager=*/true);

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
  os << "    // TODO: rst_ni tied high - the crossbar never resets "
        "(stopgap).\n";
  os << "    .rst_ni                (1'b1),\n";
  os << "    .test_i                (1'b0),\n";
  os << "    .slv_ports_req_i       (slv_req),\n";
  os << "    .slv_ports_resp_o      (slv_resp),\n";
  os << "    .mst_ports_req_o       (mst_req),\n";
  os << "    .mst_ports_resp_i      (mst_resp),\n";
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

  // TODO: once generic inputs/outputs are brought in, this should be relaxed to
  // allow non-axi4 users of generic_outputs
  auto checkUsers = [&](Operation *op) {
    for (Operation *user : op->getUsers())
      if (!isa_and_nonnull<AXI4Dialect>(user->getDialect())) {
        op->emitError("results of axi4 operations may only be used by axi4 "
                      "operations");
        failed = true;
        break;
      }
  };

  // Access windows must fit the port's address space; otherwise the derived
  // routing rule would wrap and silently mis-route.
  // TODO: drop this once an op verifier enforces windows fit the address
  // space.
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
          checkClock(sub, sub.getClock());
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
        })
        .Case<XbarOp>([&](XbarOp xbar) {
          checkClock(xbar, xbar.getClock());
          checkRegion(xbar);
          checkUsers(xbar);
          // PULP's rule_t is xbar_rule_32_t/xbar_rule_64_t; wider addresses
          // have nowhere to go.
          // TODO: drop once an address-width verifier guarantees this.
          if (cast<PortType>(xbar.getPort().getType()).getAddressWidth() > 64) {
            xbar.emitError(
                "address widths wider than 64 bits are not supported "
                "by the PULP axi_xbar address map");
            failed = true;
            return;
          }
          // PULP's address decoder requires the rules to be disjoint.
          SmallVector<AddrRule> rules = deriveXbarAddrMap(xbar);
          for (unsigned i = 0; i < rules.size(); ++i)
            for (unsigned k = i + 1; k < rules.size(); ++k)
              if (rules[i].start < ruleEndValue(rules[k]) &&
                  rules[k].start < ruleEndValue(rules[i])) {
                xbar.emitError("overlapping address windows in the crossbar "
                               "address map (master ports ")
                    << rules[i].idx << " and " << rules[k].idx
                    << "); the PULP axi_xbar requires disjoint address rules";
                failed = true;
                return;
              }
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
/// named `<prefix><token>`/`_valid`/`_ready` (see `buildChannelPortList`). Used
/// for the xbar wrapper, whose interface mirrors the internal `ChannelWires`
/// form directly rather than flattening payloads to per-field scalars.
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
    specs.push_back({portOp, isa<ManagerPortOp>(portOp), portType,
                     MappingKind::PortWires, portWires.getName().str(),
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
      edges[&*mgr.getPort().use_begin()].producer = wires;
    else
      edges[&portOp->getOpOperand(0)].consumer = wires;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PULP axi_xbar lowering
//===----------------------------------------------------------------------===//

sv::SVVerbatimModuleOp NetworkLowering::getOrCreateXbarModule(
    unsigned numUpstream, unsigned numDownstream, PortType upstreamType,
    PortType downstreamType, ArrayRef<AddrRule> rules) {
  MLIRContext *ctx = module.getContext();
  std::string shape =
      ("axi_xbar_" + Twine(numUpstream) + "u" + Twine(numDownstream) + "d_a" +
       Twine(upstreamType.getAddressWidth()) + "_d" +
       Twine(upstreamType.getDataWidth()) + "_i" +
       Twine(upstreamType.getIdWidth()) + "_o" +
       Twine(downstreamType.getIdWidth()))
          .str();

  // Two xbars share a wrapper only if they route identically: the address map
  // is baked into the wrapper text, so it is part of the dedup signature.
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

  // Port interface mirrors the ChannelWires form
  SmallVector<PortInfo> ports;
  ports.push_back(PortInfo{{StringAttr::get(ctx, "clk_i"),
                            IntegerType::get(ctx, 1), ModulePort::Input}});
  for (unsigned i = 0; i < numUpstream; ++i)
    buildChannelPortList(ctx, upstreamType, /*isManager=*/false,
                         ("sub" + Twine(i) + "_").str(), ports);
  for (unsigned j = 0; j < numDownstream; ++j)
    buildChannelPortList(ctx, downstreamType, /*isManager=*/true,
                         ("mgr" + Twine(j) + "_").str(), ports);

  std::string source = buildXbarWrapperSource(
      name, numUpstream, numDownstream, upstreamType, downstreamType, rules);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto srcOp = sv::SVVerbatimSourceOp::create(
      builder, StringAttr::get(ctx, name + "_source"), source,
      hw::OutputFileAttr::getFromFilename(ctx, name + ".sv"),
      builder.getArrayAttr({}), /*additional_files=*/nullptr,
      builder.getStringAttr(name));
  auto modOp = sv::SVVerbatimModuleOp::create(
      builder, StringAttr::get(ctx, name), ports, FlatSymbolRefAttr::get(srcOp),
      builder.getArrayAttr({}), builder.getStringAttr(name));
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

  // The address map (checkNetwork has already validated it is disjoint and
  // within a supported width).
  SmallVector<AddrRule> rules = deriveXbarAddrMap(xbar);

  auto moduleOp = getOrCreateXbarModule(numUpstream, numDownstream,
                                        upstreamType, downstreamType, rules);

  // Wire each face through the shared instance builder, exactly like a node:
  // upstream faces are subordinate-role, downstream faces manager-role, each
  // bound via the struct-grouped channel-split convention.
  SmallVector<PortGroupSpec> specs;
  for (unsigned i = 0; i < numUpstream; ++i)
    specs.push_back({xbar, /*isManager=*/false, upstreamType,
                     MappingKind::ChannelPorts, ("sub" + Twine(i) + "_").str(),
                     xbar.getClock(), "clk_i"});
  for (unsigned j = 0; j < numDownstream; ++j)
    specs.push_back({xbar, /*isManager=*/true, downstreamType,
                     MappingKind::ChannelPorts, ("mgr" + Twine(j) + "_").str(),
                     xbar.getClock(), "clk_i"});

  SmallVector<PortWires, 0> allWires;
  std::string instName = ("xbar_" + Twine(instanceCounter++)).str();
  if (failed(buildInstance(xbar, moduleOp, instName, specs, allWires)))
    return failure();

  // File the wires under their edges: an upstream interface is the consumer of
  // the operand feeding it; a downstream interface is the producer of a use.
  // `allWires` follows `specs`: upstream faces first, then downstream.
  unsigned upstreamBase = upstream.getBeginOperandIndex();
  for (unsigned i = 0; i < numUpstream; ++i)
    edges[&xbar->getOpOperand(upstreamBase + i)].consumer = allWires[i];
  for (unsigned j = 0; j < numDownstream; ++j)
    edges[downstreamUses[j]].producer = allWires[numUpstream + j];
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
  // its ops wherever they are and emit the lowered design in the same block, so
  // an enclosing module comes out well-formed (and export-verilog-able).
  SmallVector<NodeOp> nodes;
  SmallVector<XbarOp> xbars;
  module.walk([&](Operation *op) {
    if (auto node = dyn_cast<NodeOp>(op))
      nodes.push_back(node);
    else if (auto xbar = dyn_cast<XbarOp>(op))
      xbars.push_back(xbar);
  });

  Block *netBlock = nullptr;
  if (!nodes.empty())
    netBlock = nodes.front()->getBlock();
  else if (!xbars.empty())
    netBlock = xbars.front()->getBlock();
  if (netBlock) {
    if (!netBlock->empty() &&
        netBlock->back().hasTrait<OpTrait::IsTerminator>())
      builder.setInsertionPoint(&netBlock->back());
    else
      builder.setInsertionPointToEnd(netBlock);
  }

  // When the clock is an enclosing module's i1 port cast to !axi4.clock, read
  // that i1 directly rather than casting back (which export-verilog rejects).
  SmallVector<UnrealizedConversionCastOp> clockCasts;
  if (netBlock)
    for (Operation &op : *netBlock)
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op))
        if (cast.getNumOperands() == 1 &&
            isa<ClockType>(cast.getResult(0).getType()) &&
            cast.getOperand(0).getType().isInteger(1)) {
          clockCache[cast.getResult(0)] = cast.getOperand(0);
          clockCasts.push_back(cast);
        }

  for (NodeOp node : nodes)
    if (failed(lowerNode(node)))
      return failure();

  for (XbarOp xbar : xbars)
    if (failed(lowerXbar(xbar)))
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

  // The short-circuited clock casts are dead now that the network is gone.
  for (UnrealizedConversionCastOp cast : clockCasts)
    if (cast.use_empty())
      cast.erase();

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
