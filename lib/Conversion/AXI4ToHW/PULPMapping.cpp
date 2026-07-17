//===- PULPMapping.cpp - Lower AXI4 crossbars onto the PULP axi_xbar ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The PULP-specific realization of AXI4 crossbars: derives the axi_xbar address
// map, emits the SystemVerilog wrapper instantiating PULP's axi_xbar (via an
// sv.verbatim wrapper to handle parameterizations that would be messy in hw),
// and lowers axi4.xbar ops onto it.
//
//===----------------------------------------------------------------------===//

#include "AXI4ToHWInternals.h"
#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace circt;
using namespace circt::axi4;
using namespace circt::hw;
using namespace circt::AXI4ToHW;

namespace {

/// End of a rule for ordering/overlap math.
uint64_t ruleEndValue(const AddrRule &r) { return r.end == 0 ? ~0ull : r.end; }

/// PULP's single `user_t` width. The crossbar routes user unchanged, so
/// checkXbarSupported requires the manager and subordinate user widths to match
/// and this is that shared width; PULP still needs a valid >=1-bit type when
/// the interface carries no user.
unsigned pulpUserWidth(PortType pt) {
  return pt.getUserWidth() == 0 ? 1 : pt.getUserWidth();
}

/// Append the [start, end) address ranges served by one downstream endpoint of
/// a crossbar. A subordinate contributes its access windows; a chained crossbar
/// contributes the union of the ranges served by *its* downstreams,
/// recursively, so a crossbar can fan out to subordinates and further crossbars
/// at once.
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
  if (auto xbar = dyn_cast<XbarOp>(consumer)) {
    if (!visited.insert(xbar).second)
      return; // Guard against pathological cycles.
    for (OpOperand &use : xbar.getPort().getUses())
      collectDownstreamRanges(use.getOwner(), addrW, ranges, visited);
  }
}

/// Derive the crossbar's address map: one rule per address range served by each
/// downstream endpoint, indexed by master-port (use-list) order to match how
/// `lowerXbar` packs the arrays.
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
/// the integer widths `getChannelPayloadType` uses. The `user` field is
/// width-dependent and handled by the caller.
std::string svChannelFieldType(StringRef field, StringRef prefix,
                               StringRef idTy) {
  if (field == "id")
    return idTy.str();
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
  // checkXbarSupported has verified write==read id, so either getter works.
  unsigned slvId = upType.getWriteIdWidth();
  unsigned mstId = downType.getWriteIdWidth();
  // checkXbarSupported has verified the up/down user widths match.
  unsigned userW = pulpUserWidth(upType);
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
  os << "typedef logic [" << userW << "-1:0] " << p << "user_t;\n";
  os << "typedef logic [" << slvId << "-1:0] " << p << "slv_id_t;\n";
  os << "typedef logic [" << mstId << "-1:0] " << p << "mst_id_t;\n";
  // PULP-shaped req/resp/channel typedefs, used only internally to instantiate
  // axi_xbar.
  os << "`AXI_TYPEDEF_ALL(" << p << "slv, " << p << "addr_t, " << p
     << "slv_id_t, " << p << "data_t, " << p << "strb_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_ALL(" << p << "mst, " << p << "addr_t, " << p
     << "mst_id_t, " << p << "data_t, " << p << "strb_t, " << p << "user_t)\n";

  // Typedefs for the channel structs we use as wrapper ports. Their user field
  // is the port's user width, which equals the PULP-internal user_t above.
  auto emitCanonTypedef = [&](StringRef role, StringRef idTy, PortType pt,
                              const ChannelInfo &ci) {
    os << "typedef struct packed {";
    for (auto &fi : getChannelPayloadType(pt, ci.channel).getElements()) {
      StringRef fname = fi.name.getValue();
      if (fname == "user") {
        unsigned w = pt.getUserWidth();
        if (w == 0)
          continue; // no user bits on this interface
        os << " logic [" << w << "-1:0] user;";
        continue;
      }
      os << " " << svChannelFieldType(fname, p, idTy) << " " << fname << ";";
    }
    os << " } " << p << role << "_" << ci.token << "_t;\n";
  };
  std::string subId = (p + "slv_id_t");
  std::string mstIdTy = (p + "mst_id_t");
  for (const ChannelInfo &ci : kChannelInfos)
    emitCanonTypedef("sub", subId, upType, ci);
  for (const ChannelInfo &ci : kChannelInfos)
    emitCanonTypedef("mgr", mstIdTy, downType, ci);
  os << "\n";

  // Mirror the representation of channel payloads that we use between
  // components (matching buildChannelPortList).
  SmallVector<std::string> portDecls;
  portDecls.push_back("  input  logic clk_i");
  portDecls.push_back("  input  logic rst_ni");
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

  // Bridge each face's canonical channel ports to the PULP req/resp structs.
  // The canonical user width equals PULP's user_t, so user copies straight
  // across (tied off when the interface carries no user). PULP's atop is tied
  // to 0.
  auto payloadPattern = [&](StringRef lhsExpr, StringRef rhsExpr, PortType pt,
                            const ChannelInfo &ci, bool toPulp) -> std::string {
    unsigned userW = pt.getUserWidth();
    std::string s = "'{";
    bool first = true;
    for (auto &fi : getChannelPayloadType(pt, ci.channel).getElements()) {
      StringRef fname = fi.name.getValue();
      if (fname == "user")
        continue; // handled below (absent from the PULP struct when width 0)
      if (!first)
        s += ", ";
      first = false;
      s += (fname + ": " + rhsExpr + "." + fname).str();
    }
    if (toPulp) {
      if (ci.channel == AXI4Channel::AW)
        s += ", atop: '0";
      s += userW == 0 ? ", user: '0" : (", user: " + rhsExpr + ".user").str();
    } else if (userW != 0) {
      s += (", user: " + rhsExpr + ".user").str();
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
        // Wrapper drives the port from PULP.
        os << payloadPattern(port, pvVar + "." + tok, pt, ci, /*toPulp=*/false);
        os << "  assign " << port << "_valid = " << pvVar << "." << tok
           << "_valid;\n";
        os << "  assign " << rdyVar << "." << tok << "_ready = " << port
           << "_ready;\n";
      } else {
        // Wrapper receives the port into PULP.
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

  // Instantiate the real crossbar; test/default-port controls tied off.
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
  os << "    .rst_ni                (rst_ni),\n";
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

/// Emit the shared body of a symmetric (cut/cdc-like) PULP wrapper into `os`:
/// the file header, name-prefixed typedefs, boundary channel structs, the
/// module port list (the given clock/reset input ports, then the sub0/mgr0
/// faces), the internal PULP req/resp structs, and the boundary<->PULP bridges.
/// Leaves `os` inside the open module body for the caller to append the IP
/// instantiation and `endmodule`. `pt` is the (symmetric) port type; the write
/// and read ID widths are kept independent (neither axi_cut nor axi_cdc inspects
/// them). `ipName` names the wrapped IP in the header comment; `clkRstPorts` are
/// the leading clock/reset port declarations.
void emitSymmetricWrapperBody(llvm::raw_string_ostream &os, StringRef name,
                              PortType pt, StringRef ipName,
                              ArrayRef<StringRef> clkRstPorts) {
  unsigned addrW = pt.getAddressWidth();
  unsigned dataW = pt.getDataWidth();
  unsigned widW = pt.getWriteIdWidth();
  unsigned ridW = pt.getReadIdWidth();
  unsigned userW = pulpUserWidth(pt);
  std::string p = (name + "_").str();

  // The id type name for one channel's payload: write path (AW, B) uses the
  // write id, read path (AR, R) the read id.
  auto idTyFor = [&](const ChannelInfo &ci) -> std::string {
    bool isRead = ci.channel == AXI4Channel::AR || ci.channel == AXI4Channel::R;
    return (p + (isRead ? "rid_t" : "wid_t"));
  };

  os << "// Generated by --lower-axi4-to-hw: wrapper instantiating PULP "
     << ipName << ".\n";
  os << "`include \"axi/typedef.svh\"\n\n";

  // AXI typedefs at file scope, name-prefixed to stay collision-free.
  os << "typedef logic [" << addrW << "-1:0] " << p << "addr_t;\n";
  os << "typedef logic [" << dataW << "-1:0] " << p << "data_t;\n";
  os << "typedef logic [" << dataW << "/8-1:0] " << p << "strb_t;\n";
  os << "typedef logic [" << userW << "-1:0] " << p << "user_t;\n";
  os << "typedef logic [" << widW << "-1:0] " << p << "wid_t;\n";
  os << "typedef logic [" << ridW << "-1:0] " << p << "rid_t;\n";
  // PULP-shaped channel/req/resp typedefs, built per channel so the write and
  // read ID widths stay independent. Used only internally to instantiate
  // axi_cut.
  os << "`AXI_TYPEDEF_AW_CHAN_T(" << p << "aw_chan_t, " << p << "addr_t, " << p
     << "wid_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_W_CHAN_T(" << p << "w_chan_t, " << p << "data_t, " << p
     << "strb_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_B_CHAN_T(" << p << "b_chan_t, " << p << "wid_t, " << p
     << "user_t)\n";
  os << "`AXI_TYPEDEF_AR_CHAN_T(" << p << "ar_chan_t, " << p << "addr_t, " << p
     << "rid_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_R_CHAN_T(" << p << "r_chan_t, " << p << "data_t, " << p
     << "rid_t, " << p << "user_t)\n";
  os << "`AXI_TYPEDEF_REQ_T(" << p << "axi_req_t, " << p << "aw_chan_t, " << p
     << "w_chan_t, " << p << "ar_chan_t)\n";
  os << "`AXI_TYPEDEF_RESP_T(" << p << "axi_resp_t, " << p << "b_chan_t, " << p
     << "r_chan_t)\n";

  // Typedefs for the channel structs we use as wrapper ports. Their user field
  // is the port's user width, which equals the PULP-internal user_t above.
  auto emitCanonTypedef = [&](StringRef role, const ChannelInfo &ci) {
    os << "typedef struct packed {";
    for (auto &fi : getChannelPayloadType(pt, ci.channel).getElements()) {
      StringRef fname = fi.name.getValue();
      if (fname == "user") {
        unsigned w = pt.getUserWidth();
        if (w == 0)
          continue; // no user bits on this interface
        os << " logic [" << w << "-1:0] user;";
        continue;
      }
      os << " " << svChannelFieldType(fname, p, idTyFor(ci)) << " " << fname
         << ";";
    }
    os << " } " << p << role << "_" << ci.token << "_t;\n";
  };
  for (const ChannelInfo &ci : kChannelInfos)
    emitCanonTypedef("sub", ci);
  for (const ChannelInfo &ci : kChannelInfos)
    emitCanonTypedef("mgr", ci);
  os << "\n";

  // Mirror the representation of channel payloads that we use between
  // components (matching buildChannelPortList).
  SmallVector<std::string> portDecls;
  for (StringRef port : clkRstPorts)
    portDecls.push_back(port.str());
  auto emitFacePorts = [&](StringRef role, bool isManager) {
    for (const ChannelInfo &ci : kChannelInfos) {
      bool payloadDrivenByModule = (isManager == ci.isRequest);
      StringRef fwd = payloadDrivenByModule ? "output" : "input ";
      StringRef rev = payloadDrivenByModule ? "input " : "output";
      std::string base = (role + "0_" + ci.token).str();
      std::string ty = (p + role + "_" + ci.token + "_t").str();
      portDecls.push_back(("  " + fwd + " " + ty + " " + base).str());
      portDecls.push_back(("  " + fwd + " logic " + base + "_valid").str());
      portDecls.push_back(("  " + rev + " logic " + base + "_ready").str());
    }
  };
  emitFacePorts("sub", /*isManager=*/false);
  emitFacePorts("mgr", /*isManager=*/true);
  os << "module " << name << " (\n";
  os << llvm::join(portDecls, ",\n") << "\n";
  os << ");\n";

  // Internal PULP-shaped req/resp structs: the boundary ports bridge to these,
  // and axi_cut is wired directly to them.
  os << "  " << p << "axi_req_t  slv_req;\n";
  os << "  " << p << "axi_resp_t slv_resp;\n";
  os << "  " << p << "axi_req_t  mst_req;\n";
  os << "  " << p << "axi_resp_t mst_resp;\n";

  // Bridge each face's canonical channel ports to the PULP req/resp structs.
  // The canonical user width equals PULP's user_t, so user copies straight
  // across (tied off when the interface carries no user). PULP's atop is tied
  // to 0.
  auto payloadPattern = [&](StringRef lhsExpr, StringRef rhsExpr,
                            const ChannelInfo &ci, bool toPulp) -> std::string {
    unsigned uW = pt.getUserWidth();
    std::string s = "'{";
    bool first = true;
    for (auto &fi : getChannelPayloadType(pt, ci.channel).getElements()) {
      StringRef fname = fi.name.getValue();
      if (fname == "user")
        continue; // handled below (absent from the PULP struct when width 0)
      if (!first)
        s += ", ";
      first = false;
      s += (fname + ": " + rhsExpr + "." + fname).str();
    }
    if (toPulp) {
      if (ci.channel == AXI4Channel::AW)
        s += ", atop: '0";
      s += uW == 0 ? ", user: '0" : (", user: " + rhsExpr + ".user").str();
    } else if (uW != 0) {
      s += (", user: " + rhsExpr + ".user").str();
    }
    s += "}";
    return ("  assign " + lhsExpr + " = " + s + ";\n").str();
  };
  auto emitFaceBridge = [&](StringRef role, bool isManager) {
    StringRef reqVar = isManager ? "mst_req" : "slv_req";
    StringRef respVar = isManager ? "mst_resp" : "slv_resp";
    for (const ChannelInfo &ci : kChannelInfos) {
      bool payloadDrivenByModule = (isManager == ci.isRequest);
      std::string tok = ci.token.str();
      std::string port = (role + "0_" + tok).str();
      // payload+valid live in the req struct for request channels, the resp
      // struct for response channels; ready lives in the other one.
      StringRef pvVar = ci.isRequest ? reqVar : respVar;
      StringRef rdyVar = ci.isRequest ? respVar : reqVar;
      if (payloadDrivenByModule) {
        // Wrapper drives the port from PULP.
        os << payloadPattern(port, (pvVar + "." + tok).str(), ci,
                             /*toPulp=*/false);
        os << "  assign " << port << "_valid = " << pvVar << "." << tok
           << "_valid;\n";
        os << "  assign " << rdyVar << "." << tok << "_ready = " << port
           << "_ready;\n";
      } else {
        // Wrapper receives the port into PULP.
        os << payloadPattern((pvVar + "." + tok).str(), port, ci,
                             /*toPulp=*/true);
        os << "  assign " << pvVar << "." << tok << "_valid = " << port
           << "_valid;\n";
        os << "  assign " << port << "_ready = " << rdyVar << "." << tok
           << "_ready;\n";
      }
    }
  };
  emitFaceBridge("sub", /*isManager=*/false);
  emitFaceBridge("mgr", /*isManager=*/true);
}

/// SystemVerilog wrapper for one cut configuration: a symmetric register slice
/// via PULP's axi_cut, in a single clock/reset domain.
std::string buildCutWrapperSource(StringRef name, PortType pt) {
  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapperBody(
      os, name, pt, "axi_cut",
      {"  input  logic clk_i", "  input  logic rst_ni"});
  std::string p = (name + "_").str();

  // Instantiate the register slice; no bypass.
  os << "  axi_cut #(\n";
  os << "    .Bypass     (1'b0),\n";
  os << "    .aw_chan_t  (" << p << "aw_chan_t),\n";
  os << "    .w_chan_t   (" << p << "w_chan_t),\n";
  os << "    .b_chan_t   (" << p << "b_chan_t),\n";
  os << "    .ar_chan_t  (" << p << "ar_chan_t),\n";
  os << "    .r_chan_t   (" << p << "r_chan_t),\n";
  os << "    .axi_req_t  (" << p << "axi_req_t),\n";
  os << "    .axi_resp_t (" << p << "axi_resp_t)\n";
  os << "  ) i_cut (\n";
  os << "    .clk_i      (clk_i),\n";
  os << "    .rst_ni     (rst_ni),\n";
  os << "    .slv_req_i  (slv_req),\n";
  os << "    .slv_resp_o (slv_resp),\n";
  os << "    .mst_req_o  (mst_req),\n";
  os << "    .mst_resp_i (mst_resp)\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

/// SystemVerilog wrapper for one cdc configuration: a symmetric clock-domain
/// crossing via PULP's axi_cdc. The upstream face is clocked by the source
/// domain, the downstream face by the destination domain. FIFO depth and
/// synchronizer stages are fixed at the axi_cdc defaults (LogDepth 1,
/// SyncStages 2); the axi4.cdc op carries no knobs for them yet.
std::string buildCdcWrapperSource(StringRef name, PortType pt) {
  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapperBody(os, name, pt, "axi_cdc",
                           {"  input  logic src_clk_i",
                            "  input  logic src_rst_ni",
                            "  input  logic dst_clk_i",
                            "  input  logic dst_rst_ni"});
  std::string p = (name + "_").str();

  // Instantiate the clock-domain-crossing FIFO. slv side = source domain,
  // mst side = destination domain.
  os << "  axi_cdc #(\n";
  os << "    .aw_chan_t  (" << p << "aw_chan_t),\n";
  os << "    .w_chan_t   (" << p << "w_chan_t),\n";
  os << "    .b_chan_t   (" << p << "b_chan_t),\n";
  os << "    .ar_chan_t  (" << p << "ar_chan_t),\n";
  os << "    .r_chan_t   (" << p << "r_chan_t),\n";
  os << "    .axi_req_t  (" << p << "axi_req_t),\n";
  os << "    .axi_resp_t (" << p << "axi_resp_t),\n";
  os << "    .LogDepth   (1),\n";
  os << "    .SyncStages (2)\n";
  os << "  ) i_cdc (\n";
  os << "    .src_clk_i  (src_clk_i),\n";
  os << "    .src_rst_ni (src_rst_ni),\n";
  os << "    .src_req_i  (slv_req),\n";
  os << "    .src_resp_o (slv_resp),\n";
  os << "    .dst_clk_i  (dst_clk_i),\n";
  os << "    .dst_rst_ni (dst_rst_ni),\n";
  os << "    .dst_req_o  (mst_req),\n";
  os << "    .dst_resp_i (mst_resp)\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

} // namespace

namespace circt {
namespace AXI4ToHW {

/// Reject crossbars the PULP axi_xbar backend cannot represent.
LogicalResult checkXbarSupported(XbarOp xbar) {
  auto upType = cast<PortType>(xbar.getUpstream().front().getType());
  auto downType = cast<PortType>(xbar.getPort().getType());

  // PULP's rule_t is xbar_rule_32_t/xbar_rule_64_t; wider addresses have
  // nowhere to go.
  if (downType.getAddressWidth() > 64)
    return xbar.emitError("address widths wider than 64 bits are not supported "
                          "by the PULP axi_xbar address map");

  // PULP's slv_id_t/mst_id_t is a single width per side, so the write and read
  // ID widths must match.
  for (auto [pt, side] :
       {std::make_pair(upType, "upstream"), std::make_pair(downType, "master")})
    if (pt.getWriteIdWidth() != pt.getReadIdWidth())
      return xbar.emitError(
                 "the PULP axi_xbar uses a single ID width per side, "
                 "so the ")
             << side << " write ID width (" << pt.getWriteIdWidth()
             << ") and read ID width (" << pt.getReadIdWidth()
             << ") must match";

  // PULP routes the user sideband through unchanged over a single user_t, so
  // the manager and subordinate user widths must be equal.
  if (upType.getUserWidth() != downType.getUserWidth())
    return xbar.emitError("the PULP axi_xbar routes user unchanged over a "
                          "single user width, so the upstream user width (")
           << upType.getUserWidth() << ") and master user width ("
           << downType.getUserWidth() << ") must match";

  // PULP's address decoder requires the rules to be disjoint.
  SmallVector<AddrRule> rules = deriveXbarAddrMap(xbar);
  for (unsigned i = 0; i < rules.size(); ++i)
    for (unsigned k = i + 1; k < rules.size(); ++k)
      if (rules[i].start < ruleEndValue(rules[k]) &&
          rules[k].start < ruleEndValue(rules[i]))
        return xbar.emitError("overlapping address windows in the crossbar "
                              "address map (master ports ")
               << rules[i].idx << " and " << rules[k].idx
               << "); the PULP axi_xbar requires disjoint address rules";
  return success();
}

sv::SVVerbatimModuleOp NetworkLowering::getOrCreateXbarModule(
    unsigned numUpstream, unsigned numDownstream, PortType upstreamType,
    PortType downstreamType, ArrayRef<AddrRule> rules) {
  MLIRContext *ctx = module.getContext();
  std::string shape =
      ("axi_xbar_" + Twine(numUpstream) + "u" + Twine(numDownstream) + "d_a" +
       Twine(upstreamType.getAddressWidth()) + "_d" +
       Twine(upstreamType.getDataWidth()) + "_i" +
       Twine(upstreamType.getWriteIdWidth()) + "_o" +
       Twine(downstreamType.getWriteIdWidth()))
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

  // Port interface mirrors the ChannelWires form.
  SmallVector<PortInfo> ports;
  Type i1 = IntegerType::get(ctx, 1);
  ports.push_back(
      PortInfo{{StringAttr::get(ctx, "clk_i"), i1, ModulePort::Input}});
  ports.push_back(
      PortInfo{{StringAttr::get(ctx, "rst_ni"), i1, ModulePort::Input}});
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

sv::SVVerbatimModuleOp NetworkLowering::getOrCreateCutModule(PortType pt) {
  MLIRContext *ctx = module.getContext();
  std::string shape =
      ("axi_cut_a" + Twine(pt.getAddressWidth()) + "_d" +
       Twine(pt.getDataWidth()) + "_iw" + Twine(pt.getWriteIdWidth()) + "_ir" +
       Twine(pt.getReadIdWidth()) + "_u" + Twine(pt.getUserWidth()))
          .str();
  if (auto existing = cutWrappers.lookup(shape))
    return existing;

  // Name after the shape; disambiguate against unrelated symbols.
  std::string name = shape;
  for (unsigned n = 0; module.lookupSymbol(name); ++n)
    name = (shape + "_" + Twine(n)).str();

  // Port interface mirrors the ChannelWires form.
  SmallVector<PortInfo> ports;
  Type i1 = IntegerType::get(ctx, 1);
  ports.push_back(
      PortInfo{{StringAttr::get(ctx, "clk_i"), i1, ModulePort::Input}});
  ports.push_back(
      PortInfo{{StringAttr::get(ctx, "rst_ni"), i1, ModulePort::Input}});
  buildChannelPortList(ctx, pt, /*isManager=*/false, "sub0_", ports);
  buildChannelPortList(ctx, pt, /*isManager=*/true, "mgr0_", ports);

  std::string source = buildCutWrapperSource(name, pt);

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
  cutWrappers[shape] = modOp;
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
                     xbar.getClock(), "clk_i", xbar.getReset(), "rst_ni"});
  for (unsigned j = 0; j < numDownstream; ++j)
    specs.push_back({xbar, /*isManager=*/true, downstreamType,
                     MappingKind::ChannelPorts, ("mgr" + Twine(j) + "_").str(),
                     xbar.getClock(), "clk_i", xbar.getReset(), "rst_ni"});

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

LogicalResult NetworkLowering::lowerCut(CutOp cut) {
  builder.setLoc(cut.getLoc());

  // TypesMatchWith forces the upstream and downstream port types equal, and
  // checkNetwork has verified the result feeds exactly one downstream port.
  auto portTy = cast<PortType>(cut.getUpstream().getType());
  OpOperand &downstreamUse = *cut.getDownstream().getUses().begin();

  auto moduleOp = getOrCreateCutModule(portTy);

  // Wire both faces through the shared instance builder, exactly like a node:
  // the upstream face is subordinate-role, the downstream face manager-role,
  // each bound via the struct-grouped channel-split convention.
  SmallVector<PortGroupSpec> specs;
  specs.push_back({cut, /*isManager=*/false, portTy, MappingKind::ChannelPorts,
                   "sub0_", cut.getClock(), "clk_i", cut.getReset(), "rst_ni"});
  specs.push_back({cut, /*isManager=*/true, portTy, MappingKind::ChannelPorts,
                   "mgr0_", cut.getClock(), "clk_i", cut.getReset(), "rst_ni"});

  SmallVector<PortWires, 0> allWires;
  std::string instName = ("cut_" + Twine(instanceCounter++)).str();
  if (failed(buildInstance(cut, moduleOp, instName, specs, allWires)))
    return failure();

  // File the wires under their edges: the upstream face is the consumer of the
  // operand feeding it; the downstream face is the producer of the result use.
  edges[&cut.getUpstreamMutable()].consumer = allWires[0];
  edges[&downstreamUse].producer = allWires[1];
  return success();
}

sv::SVVerbatimModuleOp NetworkLowering::getOrCreateCdcModule(PortType pt) {
  MLIRContext *ctx = module.getContext();
  std::string shape =
      ("axi_cdc_a" + Twine(pt.getAddressWidth()) + "_d" +
       Twine(pt.getDataWidth()) + "_iw" + Twine(pt.getWriteIdWidth()) + "_ir" +
       Twine(pt.getReadIdWidth()) + "_u" + Twine(pt.getUserWidth()))
          .str();
  if (auto existing = cdcWrappers.lookup(shape))
    return existing;

  // Name after the shape; disambiguate against unrelated symbols.
  std::string name = shape;
  for (unsigned n = 0; module.lookupSymbol(name); ++n)
    name = (shape + "_" + Twine(n)).str();

  // Port interface mirrors the ChannelWires form, with a source and a
  // destination clock/reset for the two domains.
  SmallVector<PortInfo> ports;
  Type i1 = IntegerType::get(ctx, 1);
  for (StringRef clkRst : {"src_clk_i", "src_rst_ni", "dst_clk_i", "dst_rst_ni"})
    ports.push_back(
        PortInfo{{StringAttr::get(ctx, clkRst), i1, ModulePort::Input}});
  buildChannelPortList(ctx, pt, /*isManager=*/false, "sub0_", ports);
  buildChannelPortList(ctx, pt, /*isManager=*/true, "mgr0_", ports);

  std::string source = buildCdcWrapperSource(name, pt);

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
  cdcWrappers[shape] = modOp;
  return modOp;
}

LogicalResult NetworkLowering::lowerCdc(CDCOp cdc) {
  builder.setLoc(cdc.getLoc());

  // TypesMatchWith forces the upstream and downstream port types equal, and
  // checkNetwork has verified the result feeds exactly one downstream port.
  auto portTy = cast<PortType>(cdc.getUpstream().getType());
  OpOperand &downstreamUse = *cdc.getDownstream().getUses().begin();

  auto moduleOp = getOrCreateCdcModule(portTy);

  // Wire both faces through the shared instance builder. The upstream face is
  // subordinate-role in the source clock/reset domain; the downstream face is
  // manager-role in the destination domain.
  SmallVector<PortGroupSpec> specs;
  specs.push_back({cdc, /*isManager=*/false, portTy, MappingKind::ChannelPorts,
                   "sub0_", cdc.getUpstreamClock(), "src_clk_i",
                   cdc.getUpstreamReset(), "src_rst_ni"});
  specs.push_back({cdc, /*isManager=*/true, portTy, MappingKind::ChannelPorts,
                   "mgr0_", cdc.getDownstreamClock(), "dst_clk_i",
                   cdc.getDownstreamReset(), "dst_rst_ni"});

  SmallVector<PortWires, 0> allWires;
  std::string instName = ("cdc_" + Twine(instanceCounter++)).str();
  if (failed(buildInstance(cdc, moduleOp, instName, specs, allWires)))
    return failure();

  // File the wires under their edges: the upstream face is the consumer of the
  // operand feeding it; the downstream face is the producer of the result use.
  edges[&cdc.getUpstreamMutable()].consumer = allWires[0];
  edges[&downstreamUse].producer = allWires[1];
  return success();
}

} // namespace AXI4ToHW
} // namespace circt
