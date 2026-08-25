//===- PULPMapping.cpp - Map AXI4 components onto the PULP AXI library ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Maps AXI4 components onto the PULP Platform AXI library. PULP's components
// are heavily parameterized on config and address mapping, so the wrappers
// implementing their external modules are emitted as verbatim source rather
// than built out of HW ops.
//
//===----------------------------------------------------------------------===//

#include "AXI4ToHWInternals.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace axi4;
using namespace mlir;
using namespace circt::AXI4ToHW;

/// The width of PULP's `user_t`. A port with no user field still gets a bit,
/// since a zero-width typedef is not legal SystemVerilog.
static unsigned pulpUserWidth(PortType port) {
  return std::max(port.getUserWidth(), 1u);
}

/// Report a port carrying no ID bits on a channel, which no wrapper can
/// express: they type their ID fields through a typedef, and a zero-width
/// typedef is not legal SystemVerilog.
static LogicalResult checkPulpIdsPresent(Operation *op) {
  auto check = [&](Value value, const Twine &portDesc) -> LogicalResult {
    auto port = cast<PortType>(value.getType());
    if (port.getWriteIdWidth() != 0 && port.getReadIdWidth() != 0)
      return success();
    return op->emitOpError() << "cannot be lowered to PULP because its "
                             << portDesc << " has a zero-width "
                             << (port.getWriteIdWidth() == 0 ? "write" : "read")
                             << " ID, which PULP cannot express";
  };

  unsigned upstream = 0, downstream = 0;
  for (Value operand : op->getOperands())
    if (isa<PortType>(operand.getType()))
      if (failed(check(operand, "upstream port #" + Twine(upstream++))))
        return failure();
  for (Value result : op->getResults())
    if (isa<PortType>(result.getType()))
      if (failed(check(result, "downstream port #" + Twine(downstream++))))
        return failure();
  return success();
}

/// Report a port whose write and read ID widths differ, which the PULP IPs that
/// route over one ID width cannot express. `side` names the side the port is
/// on, for the IPs that have a width per side rather than one throughout.
static LogicalResult checkPulpIdWidths(Operation *op, StringRef ip,
                                       PortType port, StringRef side = "") {
  if (port.getWriteIdWidth() == port.getReadIdWidth())
    return success();
  std::string sided = side.empty() ? "" : (side + " ").str();
  return op->emitOpError() << "cannot be lowered to a PULP " << ip
                           << ", which uses a single ID width"
                           << (side.empty() ? "" : " per side")
                           << ", because its " << sided << "write ID width ("
                           << port.getWriteIdWidth() << ") and read ID width ("
                           << port.getReadIdWidth() << ") differ";
}

/// Report a downstream port not exactly as wide as the tag PULP adds to route
/// `numUpstream` managers' transactions -  op verifier only checks it is at
/// least that size.
static LogicalResult checkPulpTagWidth(Operation *op, StringRef ip,
                                       PortType upstream, unsigned numUpstream,
                                       PortType downstream) {
  uint32_t idBits = llvm::Log2_64_Ceil(numUpstream);
  if (downstream.getWriteIdWidth() == upstream.getWriteIdWidth() + idBits)
    return success();
  return op->emitOpError() << "cannot be lowered to a PULP " << ip
                           << ", which widens IDs by exactly the " << idBits
                           << " bits needed to tag " << numUpstream
                           << (numUpstream == 1 ? " manager" : " managers")
                           << ", so its downstream ID width must be "
                           << upstream.getWriteIdWidth() + idBits << ", not "
                           << downstream.getWriteIdWidth();
}

/// Report the crossbars PULP's axi_xbar cannot express.
static LogicalResult checkPulpXbarSupported(XbarOp xbar) {
  auto upstream = cast<PortType>(xbar.getUpstream().front().getType());
  if (failed(checkPulpIdWidths(xbar, "axi_xbar", upstream, "upstream")))
    return failure();

  auto downstream = cast<PortType>(xbar.getDownstream().front().getType());
  if (failed(checkPulpIdWidths(xbar, "axi_xbar", downstream, "downstream")))
    return failure();

  // The op verifier lets the downstream ports widen their IDs independently,
  // but PULP drives them all from one mst_id_t.
  for (auto [index, value] : llvm::enumerate(xbar.getDownstream().drop_front()))
    if (auto other = cast<PortType>(value.getType());
        other.getWriteIdWidth() != downstream.getWriteIdWidth() ||
        other.getReadIdWidth() != downstream.getReadIdWidth())
      return xbar.emitOpError()
             << "cannot be lowered to a PULP axi_xbar, which uses one ID width "
                "for every downstream port, because downstream port #"
             << index + 1 << "'s ID width (" << other.getWriteIdWidth()
             << ") differs from downstream port #0's ("
             << downstream.getWriteIdWidth() << ")";

  return checkPulpTagWidth(xbar, "axi_xbar", upstream,
                           xbar.getUpstream().size(), downstream);
}

/// Map from field to PULP typedef
static StringRef pulpFieldType(StringRef field) {
  return StringSwitch<StringRef>(field)
      .Case("len", "axi_pkg::len_t")
      .Case("size", "axi_pkg::size_t")
      .Case("burst", "axi_pkg::burst_t")
      .Case("cache", "axi_pkg::cache_t")
      .Case("prot", "axi_pkg::prot_t")
      .Case("qos", "axi_pkg::qos_t")
      .Case("region", "axi_pkg::region_t")
      .Case("resp", "axi_pkg::resp_t")
      // The rest have no axi_pkg type, so the caller names them itself
      .Default("");
}

/// Emit the typedef of the payload struct `port`'s `info` channel lowers to,
/// named `<prefix><role>_<channel>_t`. `fieldType` names the fields that have
/// no `axi_pkg` type of their own and are not a plain bit.
static void
emitPayloadStruct(llvm::raw_ostream &os, StringRef prefix, StringRef role,
                  PortType port, const ChannelInfo &info,
                  llvm::function_ref<std::string(StringRef)> fieldType) {
  os << "typedef struct packed {";
  for (auto field :
       cast<hw::StructType>(getChannelPayloadType(port, info.channel))
           .getElements()) {
    StringRef fieldName = field.name.getValue();
    unsigned width = hw::getBitWidth(field.type);
    // ExportVerilog drops zero-width fields from the port's struct, so an empty
    // user field is dropped here too to keep the two layouts identical.
    if (fieldName == "user" && width == 0)
      continue;
    StringRef named = pulpFieldType(fieldName);
    if (!named.empty())
      os << " " << named << " " << fieldName << ";";
    else if (width == 1)
      os << " logic " << fieldName << ";";
    else
      os << " " << fieldType(fieldName) << " " << fieldName << ";";
  }
  os << " } " << prefix << role << "_" << info.name << "_t;\n";
}

/// Append the declarations of the ports one face of a wrapper carries,
/// mirroring the signals the external module explodes into.
static void emitFacePorts(SmallVectorImpl<std::string> &ports, StringRef prefix,
                          StringRef role, unsigned index, bool isManager) {
  for (const ChannelInfo &info : kChannels) {
    bool drives = isManager == info.isRequest;
    StringRef forward = drives ? "output" : "input ";
    StringRef reverse = drives ? "input " : "output";
    std::string base = (role + Twine(index) + "_" + info.name).str();
    std::string type = (prefix + role + "_" + info.name + "_t").str();
    ports.push_back(("  " + forward + " " + type + " " + base).str());
    ports.push_back(("  " + forward + " logic " + base + "valid").str());
    ports.push_back(("  " + reverse + " logic " + base + "ready").str());
  }
}

/// Emit the assignments bridging one face's ports to the PULP `req` and `resp`
/// structs it drives. PULP's atop is tied off, and if PULP has a user field
/// that the port does not, the bridge drives it with zeroes.
static void emitFaceBridge(llvm::raw_ostream &os, PortType port, StringRef role,
                           unsigned index, bool isManager, StringRef req,
                           StringRef resp) {
  auto payloadAssign = [&](const Twine &lhs, const Twine &rhs,
                           const ChannelInfo &info, bool toPulp) {
    SmallVector<std::string> fields;
    for (auto field :
         cast<hw::StructType>(getChannelPayloadType(port, info.channel))
             .getElements()) {
      StringRef fieldName = field.name.getValue();
      // A zero-width user field is absent from the port's struct, so there is
      // nothing to read or drive on that side.
      if (fieldName == "user" && port.getUserWidth() == 0)
        continue;
      fields.push_back((fieldName + ": " + rhs + "." + fieldName).str());
    }
    if (toPulp) {
      if (info.channel == AXI4Channel::AW)
        fields.push_back("atop: '0");
      // PULP's user_t is a bit wide even when the port carries no user.
      if (port.getUserWidth() == 0)
        fields.push_back("user: '0");
    }
    os << "  assign " << lhs << " = '{" << llvm::join(fields, ", ") << "};\n";
  };

  for (const ChannelInfo &info : kChannels) {
    std::string wire = (role + Twine(index) + "_" + info.name).str();
    // A channel's payload and valid live in the struct its requester drives,
    // and its ready in the other one.
    StringRef payload = info.isRequest ? req : resp;
    StringRef ready = info.isRequest ? resp : req;
    if (isManager == info.isRequest) {
      payloadAssign(wire, payload + "." + info.name, info, /*toPulp=*/false);
      os << "  assign " << wire << "valid = " << payload << "." << info.name
         << "_valid;\n";
      os << "  assign " << ready << "." << info.name << "_ready = " << wire
         << "ready;\n";
    } else {
      payloadAssign(payload + "." + info.name, wire, info, /*toPulp=*/true);
      os << "  assign " << payload << "." << info.name << "_valid = " << wire
         << "valid;\n";
      os << "  assign " << wire << "ready = " << ready << "." << info.name
         << "_ready;\n";
    }
  }
}

/// The address map rules routing each window of `downstream` to the index of
/// the port carrying it. The AXI dialect has an inclusive last address, and
/// PULP's end_addr is exclusive. PULP uses an end address of 0 to indicate
/// wrapping to the end of the address space - conveniently if we get the
/// exclusive last address of an AXI window by adding 1, we also wrap round to
/// 0.
static SmallVector<std::string> pulpAddrMapRules(ValueRange downstream,
                                                 unsigned addrWidth) {
  uint64_t mask =
      addrWidth == 64 ? ~uint64_t{0} : (uint64_t{1} << addrWidth) - 1;
  SmallVector<std::string> rules;
  for (auto [index, value] : llvm::enumerate(downstream))
    for (WindowAttr window :
         cast<PortType>(value.getType()).getWindows().getWindows())
      rules.push_back(("    '{idx: " + Twine(index) +
                       ", start_addr: " + Twine(addrWidth) + "'h" +
                       llvm::utohexstr(window.getBase()) +
                       ", end_addr: " + Twine(addrWidth) + "'h" +
                       llvm::utohexstr((window.getLast() + 1) & mask) + "}")
                          .str());
  return rules;
}

/// Emit `rules` as the address map an IP decodes. A rule's addresses have to be
/// as wide as the ones being decoded, so the map carries its own struct rather
/// than an `axi_pkg` one, whose addresses are always 32 or 64 bits wide.
static void emitAddrMap(llvm::raw_ostream &os, StringRef prefix,
                        ArrayRef<std::string> rules) {
  os << "  typedef struct packed { int unsigned idx; " << prefix
     << "addr_t start_addr; " << prefix << "addr_t end_addr; } rule_t;\n";
  os << "  localparam rule_t [" << rules.size() << "-1:0] AddrMap = '{\n";
  os << llvm::join(rules, ",\n") << "\n";
  os << "  };\n";
}

/// The largest number of transactions any of `ports` keeps outstanding.
static uint32_t maxOutstanding(ValueRange ports) {
  uint32_t most = 0;
  for (Value value : ports) {
    auto port = cast<PortType>(value.getType());
    most = std::max(
        {most, port.getOutstandingWrites(), port.getOutstandingReads()});
  }
  return most;
}

/// Emit the body of a wrapper around a PULP module whose upstream
/// and downstream faces carry different ID widths. Leaves `os` inside the
/// module.
static void emitDualIdWrapper(llvm::raw_ostream &os, StringRef name,
                              StringRef ip, PortType upstream,
                              unsigned numUpstream, PortType downstream,
                              unsigned numDownstream) {
  unsigned addrWidth = upstream.getAddrWidth();
  unsigned dataWidth = upstream.getDataWidth();
  // checkPulp* methods already verified ID widths
  unsigned upstreamId = upstream.getWriteIdWidth();
  unsigned downstreamId = downstream.getWriteIdWidth();
  unsigned userWidth = pulpUserWidth(upstream);
  std::string prefix = (name + "_").str();

  os << "// Generated by --lower-axi4-to-hw=pulp-mapping: a wrapper around the "
        "PULP Platform "
     << ip << ".\n";
  os << "`include \"axi/typedef.svh\"\n\n";

  // Prefixed so several wrappers can share a compilation unit.
  os << "typedef logic [" << addrWidth << "-1:0] " << prefix << "addr_t;\n";
  os << "typedef logic [" << dataWidth << "-1:0] " << prefix << "data_t;\n";
  os << "typedef logic [" << dataWidth << "/8-1:0] " << prefix << "strb_t;\n";
  os << "typedef logic [" << userWidth << "-1:0] " << prefix << "user_t;\n";
  os << "typedef logic [" << upstreamId << "-1:0] " << prefix << "slv_id_t;\n";
  os << "typedef logic [" << downstreamId << "-1:0] " << prefix
     << "mst_id_t;\n";
  os << "`AXI_TYPEDEF_ALL(" << prefix << "slv, " << prefix << "addr_t, "
     << prefix << "slv_id_t, " << prefix << "data_t, " << prefix << "strb_t, "
     << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_ALL(" << prefix << "mst, " << prefix << "addr_t, "
     << prefix << "mst_id_t, " << prefix << "data_t, " << prefix << "strb_t, "
     << prefix << "user_t)\n\n";

  // A typedef per channel payload, matching the hw.struct the port lowers to.
  // The wrapper routes over one ID width per side.
  auto fieldType = [&](StringRef side) {
    return [&, side](StringRef field) {
      return field == "id" ? (prefix + side + "_id_t").str()
                           : (prefix + field + "_t").str();
    };
  };
  for (const ChannelInfo &info : kChannels)
    emitPayloadStruct(os, prefix, "mgr", upstream, info, fieldType("slv"));
  for (const ChannelInfo &info : kChannels)
    emitPayloadStruct(os, prefix, "sub", downstream, info, fieldType("mst"));
  os << "\n";

  // The wrapper is the subordinate to each upstream manager, and the manager
  // to each downstream subordinate.
  SmallVector<std::string> ports{"  input  logic clk_i",
                                 "  input  logic rst_ni"};
  for (unsigned i = 0; i != numUpstream; ++i)
    emitFacePorts(ports, prefix, "mgr", i, /*isManager=*/false);
  for (unsigned j = 0; j != numDownstream; ++j)
    emitFacePorts(ports, prefix, "sub", j, /*isManager=*/true);
  os << "module " << name << " (\n" << llvm::join(ports, ",\n") << "\n);\n";

  os << "  " << prefix << "slv_req_t  [" << numUpstream << "-1:0] slv_req;\n";
  os << "  " << prefix << "slv_resp_t [" << numUpstream << "-1:0] slv_resp;\n";
  os << "  " << prefix << "mst_req_t  [" << numDownstream << "-1:0] mst_req;\n";
  os << "  " << prefix << "mst_resp_t [" << numDownstream
     << "-1:0] mst_resp;\n";

  // Bridge each port group to the PULP req/resp struct of its index.
  for (unsigned i = 0; i != numUpstream; ++i)
    emitFaceBridge(os, upstream, "mgr", i, /*isManager=*/false,
                   ("slv_req[" + Twine(i) + "]").str(),
                   ("slv_resp[" + Twine(i) + "]").str());
  for (unsigned j = 0; j != numDownstream; ++j)
    emitFaceBridge(os, downstream, "sub", j, /*isManager=*/true,
                   ("mst_req[" + Twine(j) + "]").str(),
                   ("mst_resp[" + Twine(j) + "]").str());
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_xbar, with the
/// ports `xbar`'s external module lowers to.
static std::string pulpXbarSource(StringRef name, XbarOp xbar) {
  auto upstream = cast<PortType>(xbar.getUpstream().front().getType());
  auto downstream = cast<PortType>(xbar.getDownstream().front().getType());
  unsigned numUpstream = xbar.getUpstream().size();
  unsigned numDownstream = xbar.getDownstream().size();
  unsigned addrWidth = upstream.getAddrWidth();
  // checkPulpSupported has established one ID width per side.
  unsigned upstreamId = upstream.getWriteIdWidth();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitDualIdWrapper(os, name, "axi_xbar", upstream, numUpstream, downstream,
                    numDownstream);

  SmallVector<std::string> rules =
      pulpAddrMapRules(xbar.getDownstream(), addrWidth);

  // `default: '0` covers the Cfg fields later PULP versions added.
  os << "  localparam axi_pkg::xbar_cfg_t Cfg = '{\n";
  os << "    NoSlvPorts:         " << numUpstream << ",\n";
  os << "    NoMstPorts:         " << numDownstream << ",\n";
  os << "    MaxSlvTrans:        " << maxOutstanding(xbar.getUpstream())
     << ",\n";
  os << "    MaxMstTrans:        " << maxOutstanding(xbar.getDownstream())
     << ",\n";
  os << "    FallThrough:        1'b0,\n";
  os << "    LatencyMode:        axi_pkg::CUT_ALL_AX,\n";
  os << "    AxiIdWidthSlvPorts: " << upstreamId << ",\n";
  os << "    AxiIdUsedSlvPorts:  " << upstreamId << ",\n";
  os << "    UniqueIds:          1'b0,\n";
  os << "    AxiAddrWidth:       " << addrWidth << ",\n";
  os << "    AxiDataWidth:       " << upstream.getDataWidth() << ",\n";
  os << "    NoAddrRules:        " << rules.size() << ",\n";
  os << "    default:            '0\n";
  os << "  };\n";
  emitAddrMap(os, prefix, rules);

  os << "  axi_xbar #(\n";
  os << "    .Cfg           (Cfg),\n";
  os << "    .ATOPs         (1'b1),\n";
  os << "    .Connectivity  ('1),\n";
  os << "    .slv_aw_chan_t (" << prefix << "slv_aw_chan_t),\n";
  os << "    .mst_aw_chan_t (" << prefix << "mst_aw_chan_t),\n";
  os << "    .w_chan_t      (" << prefix << "slv_w_chan_t),\n";
  os << "    .slv_b_chan_t  (" << prefix << "slv_b_chan_t),\n";
  os << "    .mst_b_chan_t  (" << prefix << "mst_b_chan_t),\n";
  os << "    .slv_ar_chan_t (" << prefix << "slv_ar_chan_t),\n";
  os << "    .mst_ar_chan_t (" << prefix << "mst_ar_chan_t),\n";
  os << "    .slv_r_chan_t  (" << prefix << "slv_r_chan_t),\n";
  os << "    .mst_r_chan_t  (" << prefix << "mst_r_chan_t),\n";
  os << "    .slv_req_t     (" << prefix << "slv_req_t),\n";
  os << "    .slv_resp_t    (" << prefix << "slv_resp_t),\n";
  os << "    .mst_req_t     (" << prefix << "mst_req_t),\n";
  os << "    .mst_resp_t    (" << prefix << "mst_resp_t),\n";
  os << "    .rule_t        (rule_t)\n";
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

/// Emit the header and typedefs a wrapper named `name` around PULP's `ip`
/// needs to carry `port` on each face in `roles`: the field types, the channel
/// and req/resp types built from them, and a payload struct per face and
/// channel.
static void emitWrapperTypedefs(llvm::raw_ostream &os, StringRef name,
                                PortType port, StringRef ip,
                                ArrayRef<StringRef> roles) {
  std::string prefix = (name + "_").str();

  os << "// Generated by --lower-axi4-to-hw=pulp-mapping: a wrapper around the "
        "PULP Platform "
     << ip << ".\n";
  os << "`include \"axi/typedef.svh\"\n\n";

  // Prefixed so several wrappers can share a compilation unit.
  os << "typedef logic [" << port.getAddrWidth() << "-1:0] " << prefix
     << "addr_t;\n";
  os << "typedef logic [" << port.getDataWidth() << "-1:0] " << prefix
     << "data_t;\n";
  os << "typedef logic [" << port.getDataWidth() << "/8-1:0] " << prefix
     << "strb_t;\n";
  os << "typedef logic [" << pulpUserWidth(port) << "-1:0] " << prefix
     << "user_t;\n";
  os << "typedef logic [" << port.getWriteIdWidth() << "-1:0] " << prefix
     << "wid_t;\n";
  os << "typedef logic [" << port.getReadIdWidth() << "-1:0] " << prefix
     << "rid_t;\n";
  // Built per channel rather than with AXI_TYPEDEF_ALL, so that the write path
  // (AW, B) and the read path (AR, R) can carry their own ID widths.
  os << "`AXI_TYPEDEF_AW_CHAN_T(" << prefix << "aw_chan_t, " << prefix
     << "addr_t, " << prefix << "wid_t, " << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_W_CHAN_T(" << prefix << "w_chan_t, " << prefix
     << "data_t, " << prefix << "strb_t, " << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_B_CHAN_T(" << prefix << "b_chan_t, " << prefix
     << "wid_t, " << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_AR_CHAN_T(" << prefix << "ar_chan_t, " << prefix
     << "addr_t, " << prefix << "rid_t, " << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_R_CHAN_T(" << prefix << "r_chan_t, " << prefix
     << "data_t, " << prefix << "rid_t, " << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_REQ_T(" << prefix << "req_t, " << prefix << "aw_chan_t, "
     << prefix << "w_chan_t, " << prefix << "ar_chan_t)\n";
  os << "`AXI_TYPEDEF_RESP_T(" << prefix << "resp_t, " << prefix << "b_chan_t, "
     << prefix << "r_chan_t)\n\n";

  // A typedef per channel payload, matching the hw.struct the port lowers to.
  auto fieldType = [&](const ChannelInfo &info) {
    bool isRead =
        info.channel == AXI4Channel::AR || info.channel == AXI4Channel::R;
    return [&, isRead](StringRef field) {
      return field == "id"
                 ? (Twine(prefix) + (isRead ? "rid_t" : "wid_t")).str()
                 : (prefix + field + "_t").str();
    };
  };
  for (StringRef role : roles)
    for (const ChannelInfo &info : kChannels)
      emitPayloadStruct(os, prefix, role, port, info, fieldType(info));
  os << "\n";
}

/// Emit the body of a wrapper named `name` around PULP's `ip`, whose faces all
/// carry `port`: the typedefs, the module header taking `domainPorts` ahead of
/// one upstream face and `numDownstream` downstream ones, the PULP req/resp
/// structs, and the bridges to them. The downstream structs are an array when
/// there is more than one face. Leaves `os` inside the module, for the caller
/// to instantiate `ip` and close it.
static void emitSymmetricWrapper(llvm::raw_ostream &os, StringRef name,
                                 PortType port, StringRef ip,
                                 ArrayRef<StringRef> domainPorts,
                                 unsigned numDownstream = 1) {
  std::string prefix = (name + "_").str();
  emitWrapperTypedefs(os, name, port, ip, {"mgr", "sub"});

  // The wrapper is the subordinate to its upstream manager, and the manager to
  // its downstream subordinate.
  SmallVector<std::string> ports = llvm::map_to_vector(
      domainPorts, [](StringRef port) { return port.str(); });
  emitFacePorts(ports, prefix, "mgr", 0, /*isManager=*/false);
  for (unsigned j = 0; j != numDownstream; ++j)
    emitFacePorts(ports, prefix, "sub", j, /*isManager=*/true);
  os << "module " << name << " (\n" << llvm::join(ports, ",\n") << "\n);\n";

  std::string dimension =
      numDownstream == 1 ? "" : ("[" + Twine(numDownstream) + "-1:0] ").str();
  os << "  " << prefix << "req_t  slv_req;\n";
  os << "  " << prefix << "resp_t slv_resp;\n";
  os << "  " << prefix << "req_t  " << dimension << "mst_req;\n";
  os << "  " << prefix << "resp_t " << dimension << "mst_resp;\n";

  emitFaceBridge(os, port, "mgr", 0, /*isManager=*/false, "slv_req",
                 "slv_resp");
  for (unsigned j = 0; j != numDownstream; ++j) {
    std::string index = numDownstream == 1 ? "" : ("[" + Twine(j) + "]").str();
    emitFaceBridge(os, port, "sub", j, /*isManager=*/true, "mst_req" + index,
                   "mst_resp" + index);
  }
}

/// The typedefs `emitSymmetricWrapper` names the channel and req/resp types
/// after, as `axi_<kind>` parameter assignments for the IP instantiation.
static std::string pulpChannelParams(StringRef prefix) {
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "    .aw_chan_t  (" << prefix << "aw_chan_t),\n";
  os << "    .w_chan_t   (" << prefix << "w_chan_t),\n";
  os << "    .b_chan_t   (" << prefix << "b_chan_t),\n";
  os << "    .ar_chan_t  (" << prefix << "ar_chan_t),\n";
  os << "    .r_chan_t   (" << prefix << "r_chan_t),\n";
  os << "    .axi_req_t  (" << prefix << "req_t),\n";
  os << "    .axi_resp_t (" << prefix << "resp_t)";
  return text;
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_cut, with the
/// ports `cut`'s external module lowers to.
static std::string pulpCutSource(StringRef name, CutOp cut) {
  std::string prefix = (name + "_").str();
  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapper(os, name, cast<PortType>(cut.getUpstream().getType()),
                       "axi_cut",
                       {"  input  logic clk_i", "  input  logic rst_ni"});

  // A cut registers both directions, so it never bypasses.
  os << "  axi_cut #(\n";
  os << "    .Bypass     (1'b0),\n";
  os << pulpChannelParams(prefix) << "\n";
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

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_cdc, with the
/// ports `cdc`'s external module lowers to. The upstream face is clocked by the
/// source domain and the downstream face by the destination domain, both reset
/// by the one reset the crossing may not cross.
static std::string pulpCdcSource(StringRef name, CDCOp cdc) {
  std::string prefix = (name + "_").str();
  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapper(os, name, cast<PortType>(cdc.getUpstream().getType()),
                       "axi_cdc",
                       {"  input  logic src_clk_i", "  input  logic dst_clk_i",
                        "  input  logic rst_ni"});

  // Depth and synchronizer stages are left at the axi_cdc defaults, since the
  // op carries no knobs for them.
  os << "  axi_cdc #(\n";
  os << pulpChannelParams(prefix) << ",\n";
  os << "    .LogDepth   (1),\n";
  os << "    .SyncStages (2)\n";
  os << "  ) i_cdc (\n";
  os << "    .src_clk_i  (src_clk_i),\n";
  os << "    .src_rst_ni (rst_ni),\n";
  os << "    .src_req_i  (slv_req),\n";
  os << "    .src_resp_o (slv_resp),\n";
  os << "    .dst_clk_i  (dst_clk_i),\n";
  os << "    .dst_rst_ni (rst_ni),\n";
  os << "    .dst_req_o  (mst_req),\n";
  os << "    .dst_resp_i (mst_resp)\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

/// Report the width converters PULP's axi_dw_converter cannot express.
static LogicalResult checkPulpDWConverterSupported(DWConverterOp converter) {
  // The op verifier has established that the two sides agree on their ID
  // widths, but PULP converts over a single AxiIdWidth, so they must also agree
  // with each other.
  auto port = cast<PortType>(converter.getUpstream().getType());
  if (port.getWriteIdWidth() != port.getReadIdWidth())
    return converter.emitOpError()
           << "cannot be lowered to a PULP axi_dw_converter, which uses a "
              "single ID width, because its write ID width ("
           << port.getWriteIdWidth() << ") and read ID width ("
           << port.getReadIdWidth() << ") differ";
  return success();
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_dw_converter,
/// with the ports `converter`'s external module lowers to. The two faces differ
/// in their data and strobe widths, so unlike a cut they need a channel and
/// req/resp type each; the address, ID and user widths are shared.
static std::string pulpDWConverterSource(StringRef name,
                                         DWConverterOp converter) {
  auto upstream = cast<PortType>(converter.getUpstream().getType());
  auto downstream = cast<PortType>(converter.getDownstream().getType());
  unsigned upstreamData = upstream.getDataWidth();
  unsigned downstreamData = downstream.getDataWidth();
  // checkPulpDWConverterSupported has established one ID width.
  unsigned idWidth = upstream.getWriteIdWidth();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// Generated by --lower-axi4-to-hw=pulp-mapping: a wrapper around the "
        "PULP Platform axi_dw_converter.\n";
  os << "`include \"axi/typedef.svh\"\n\n";

  // Prefixed so several wrappers can share a compilation unit. Only the data
  // and strobe widths differ between the sides.
  os << "typedef logic [" << upstream.getAddrWidth() << "-1:0] " << prefix
     << "addr_t;\n";
  os << "typedef logic [" << idWidth << "-1:0] " << prefix << "id_t;\n";
  os << "typedef logic [" << pulpUserWidth(upstream) << "-1:0] " << prefix
     << "user_t;\n";
  os << "typedef logic [" << upstreamData << "-1:0] " << prefix
     << "slv_data_t;\n";
  os << "typedef logic [" << upstreamData << "/8-1:0] " << prefix
     << "slv_strb_t;\n";
  os << "typedef logic [" << downstreamData << "-1:0] " << prefix
     << "mst_data_t;\n";
  os << "typedef logic [" << downstreamData << "/8-1:0] " << prefix
     << "mst_strb_t;\n";
  // AW, AR and B carry no data, so both sides share them.
  os << "`AXI_TYPEDEF_AW_CHAN_T(" << prefix << "aw_chan_t, " << prefix
     << "addr_t, " << prefix << "id_t, " << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_B_CHAN_T(" << prefix << "b_chan_t, " << prefix << "id_t, "
     << prefix << "user_t)\n";
  os << "`AXI_TYPEDEF_AR_CHAN_T(" << prefix << "ar_chan_t, " << prefix
     << "addr_t, " << prefix << "id_t, " << prefix << "user_t)\n";
  for (const auto &[side, data, strb] :
       {std::tuple{"slv", prefix + "slv_data_t", prefix + "slv_strb_t"},
        std::tuple{"mst", prefix + "mst_data_t", prefix + "mst_strb_t"}}) {
    os << "`AXI_TYPEDEF_W_CHAN_T(" << prefix << side << "_w_chan_t, " << data
       << ", " << strb << ", " << prefix << "user_t)\n";
    os << "`AXI_TYPEDEF_R_CHAN_T(" << prefix << side << "_r_chan_t, " << data
       << ", " << prefix << "id_t, " << prefix << "user_t)\n";
    os << "`AXI_TYPEDEF_REQ_T(" << prefix << side << "_req_t, " << prefix
       << "aw_chan_t, " << prefix << side << "_w_chan_t, " << prefix
       << "ar_chan_t)\n";
    os << "`AXI_TYPEDEF_RESP_T(" << prefix << side << "_resp_t, " << prefix
       << "b_chan_t, " << prefix << side << "_r_chan_t)\n";
  }
  os << "\n";

  // A typedef per channel payload, matching the hw.struct the port lowers to.
  auto fieldType = [&](StringRef side) {
    return [&, side](StringRef field) {
      if (field == "data" || field == "strb")
        return (prefix + side + "_" + field + "_t").str();
      return (prefix + field + "_t").str();
    };
  };
  for (const ChannelInfo &info : kChannels)
    emitPayloadStruct(os, prefix, "mgr", upstream, info, fieldType("slv"));
  for (const ChannelInfo &info : kChannels)
    emitPayloadStruct(os, prefix, "sub", downstream, info, fieldType("mst"));
  os << "\n";

  // The converter is the subordinate to its upstream manager, and the manager
  // to its downstream subordinate.
  SmallVector<std::string> ports{"  input  logic clk_i",
                                 "  input  logic rst_ni"};
  emitFacePorts(ports, prefix, "mgr", 0, /*isManager=*/false);
  emitFacePorts(ports, prefix, "sub", 0, /*isManager=*/true);
  os << "module " << name << " (\n" << llvm::join(ports, ",\n") << "\n);\n";

  os << "  " << prefix << "slv_req_t  slv_req;\n";
  os << "  " << prefix << "slv_resp_t slv_resp;\n";
  os << "  " << prefix << "mst_req_t  mst_req;\n";
  os << "  " << prefix << "mst_resp_t mst_resp;\n";

  emitFaceBridge(os, upstream, "mgr", 0, /*isManager=*/false, "slv_req",
                 "slv_resp");
  emitFaceBridge(os, downstream, "sub", 0, /*isManager=*/true, "mst_req",
                 "mst_resp");

  // PULP keeps a tracker per read it is reassembling, on the upstream side, so
  // it needs one for every read the upstream port can have outstanding. A port
  // that never reads still needs one, since the trackers form an array.
  os << "  axi_dw_converter #(\n";
  os << "    .AxiMaxReads         ("
     << std::max(upstream.getOutstandingReads(), 1u) << "),\n";
  os << "    .AxiSlvPortDataWidth (" << upstreamData << "),\n";
  os << "    .AxiMstPortDataWidth (" << downstreamData << "),\n";
  os << "    .AxiAddrWidth        (" << upstream.getAddrWidth() << "),\n";
  os << "    .AxiIdWidth          (" << idWidth << "),\n";
  os << "    .aw_chan_t           (" << prefix << "aw_chan_t),\n";
  os << "    .mst_w_chan_t        (" << prefix << "mst_w_chan_t),\n";
  os << "    .slv_w_chan_t        (" << prefix << "slv_w_chan_t),\n";
  os << "    .b_chan_t            (" << prefix << "b_chan_t),\n";
  os << "    .ar_chan_t           (" << prefix << "ar_chan_t),\n";
  os << "    .mst_r_chan_t        (" << prefix << "mst_r_chan_t),\n";
  os << "    .slv_r_chan_t        (" << prefix << "slv_r_chan_t),\n";
  os << "    .axi_mst_req_t       (" << prefix << "mst_req_t),\n";
  os << "    .axi_mst_resp_t      (" << prefix << "mst_resp_t),\n";
  os << "    .axi_slv_req_t       (" << prefix << "slv_req_t),\n";
  os << "    .axi_slv_resp_t      (" << prefix << "slv_resp_t)\n";
  os << "  ) i_dw_converter (\n";
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

/// Report the ID width converters PULP's axi_iw_converter cannot express.
static LogicalResult checkPulpIWConverterSupported(IWConverterOp converter) {
  auto upstream = cast<PortType>(converter.getUpstream().getType());
  if (failed(checkPulpIdWidths(converter, "axi_iw_converter", upstream,
                               "upstream")))
    return failure();
  auto downstream = cast<PortType>(converter.getDownstream().getType());
  return checkPulpIdWidths(converter, "axi_iw_converter", downstream,
                           "downstream");
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_iw_converter,
/// with the ports `converter`'s external module lowers to. PULP prepends zeroes
/// to widen, and to narrow either remaps the IDs or, where they no longer fit,
/// serialises them onto shared ones.
static std::string pulpIWConverterSource(StringRef name,
                                         IWConverterOp converter) {
  auto upstream = cast<PortType>(converter.getUpstream().getType());
  auto downstream = cast<PortType>(converter.getDownstream().getType());
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitDualIdWrapper(os, name, "axi_iw_converter", upstream, /*numUpstream=*/1,
                    downstream, /*numDownstream=*/1);

  // A port keeps at most 2**its ID width transactions outstanding, so its
  // outstanding count bounds the unique IDs in flight as well as the
  // transactions per ID. PULP sizes its tables from both, and needs at least
  // one of each.
  unsigned slvTxns = std::max(maxOutstanding(converter.getUpstream()), 1u);
  unsigned mstTxns = std::max(maxOutstanding(converter.getDownstream()), 1u);

  os << "  axi_iw_converter #(\n";
  os << "    .AxiSlvPortIdWidth      (" << upstream.getWriteIdWidth() << "),\n";
  os << "    .AxiMstPortIdWidth      (" << downstream.getWriteIdWidth()
     << "),\n";
  os << "    .AxiSlvPortMaxUniqIds   (" << slvTxns << "),\n";
  os << "    .AxiSlvPortMaxTxnsPerId (" << slvTxns << "),\n";
  os << "    .AxiSlvPortMaxTxns      (" << slvTxns << "),\n";
  os << "    .AxiMstPortMaxUniqIds   (" << mstTxns << "),\n";
  os << "    .AxiMstPortMaxTxnsPerId (" << mstTxns << "),\n";
  os << "    .AxiAddrWidth           (" << upstream.getAddrWidth() << "),\n";
  os << "    .AxiDataWidth           (" << upstream.getDataWidth() << "),\n";
  os << "    .AxiUserWidth           (" << pulpUserWidth(upstream) << "),\n";
  os << "    .slv_req_t              (" << prefix << "slv_req_t),\n";
  os << "    .slv_resp_t             (" << prefix << "slv_resp_t),\n";
  os << "    .mst_req_t              (" << prefix << "mst_req_t),\n";
  os << "    .mst_resp_t             (" << prefix << "mst_resp_t)\n";
  os << "  ) i_iw_converter (\n";
  os << "    .clk_i      (clk_i),\n";
  os << "    .rst_ni     (rst_ni),\n";
  os << "    .slv_req_i  (slv_req[0]),\n";
  os << "    .slv_resp_o (slv_resp[0]),\n";
  os << "    .mst_req_o  (mst_req[0]),\n";
  os << "    .mst_resp_i (mst_resp[0])\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

/// Report the burst splitters PULP's axi_burst_splitter cannot express.
static LogicalResult checkPulpBurstSplitterSupported(BurstSplitterOp splitter) {
  auto upstream = cast<PortType>(splitter.getUpstream().getType());
  // Verifier already checked that upstream and downstream widths match
  if (upstream.getWriteIdWidth() != upstream.getReadIdWidth())
    return splitter.emitOpError()
           << "cannot be lowered to a PULP axi_burst_splitter, which uses a "
              "single ID width, because its write ID width ("
           << upstream.getWriteIdWidth() << ") and read ID width ("
           << upstream.getReadIdWidth() << ") differ";

  // PULP docs specify that the splitter doesn't support wrapping bursts
  for (WindowAttr window : upstream.getWindows().getWindows())
    for (BurstSpecAttr spec : window.getBurstSpecs().getBurstSpecs())
      if (spec.getKind() == BurstKind::Wrap)
        return splitter.emitOpError()
               << "cannot be lowered to a PULP axi_burst_splitter, which does "
                  "not support wrapping bursts, because its upstream port "
                  "issues "
               << spec;

  return success();
}

/// A SystemVerilog wrapper named `name` instantiating PULP's
/// axi_burst_splitter
static std::string pulpBurstSplitterSource(StringRef name,
                                           BurstSplitterOp splitter) {
  auto upstream = cast<PortType>(splitter.getUpstream().getType());
  // checkPulpBurstSplitterSupported has established one ID width.
  unsigned idWidth = upstream.getWriteIdWidth();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapper(os, name, upstream, "axi_burst_splitter",
                       {"  input  logic clk_i", "  input  logic rst_ni"});

  os << "  axi_burst_splitter #(\n";
  os << "    .MaxReadTxns  (" << std::max(upstream.getOutstandingReads(), 1u)
     << "),\n";
  os << "    .MaxWriteTxns (" << std::max(upstream.getOutstandingWrites(), 1u)
     << "),\n";
  os << "    .AddrWidth    (" << upstream.getAddrWidth() << "),\n";
  os << "    .DataWidth    (" << upstream.getDataWidth() << "),\n";
  os << "    .IdWidth      (" << idWidth << "),\n";
  os << "    .UserWidth    (" << pulpUserWidth(upstream) << "),\n";
  os << "    .axi_req_t    (" << prefix << "req_t),\n";
  os << "    .axi_resp_t   (" << prefix << "resp_t)\n";
  os << "  ) i_burst_splitter (\n";
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

/// Report the burst unwrappers PULP's axi_burst_unwrap cannot express.
static LogicalResult
checkPulpBurstUnwrapperSupported(BurstUnwrapperOp unwrapper) {
  auto upstream = cast<PortType>(unwrapper.getUpstream().getType());
  if (failed(checkPulpIdWidths(unwrapper, "axi_burst_unwrap", upstream)))
    return failure();

  // PULP computes a wrapping burst's total size in 11 bits, so 2048 bytes of it
  // truncate to zero, taking the wrap boundary derived from them along.
  unsigned bytesPerBeat = upstream.getDataWidth() / 8;
  for (WindowAttr window : upstream.getWindows().getWindows())
    for (BurstSpecAttr spec : window.getBurstSpecs().getBurstSpecs()) {
      if (spec.getKind() != BurstKind::Wrap)
        continue;
      uint64_t bytes = uint64_t{spec.getLen()} * bytesPerBeat;
      if (bytes > 2047)
        return unwrapper.emitOpError()
               << "cannot be lowered to a PULP axi_burst_unwrap, which "
                  "computes "
                  "a wrapping burst's total size in 11 bits and so supports at "
                  "most 2047 bytes, because its upstream port issues "
               << spec << " over " << upstream.getDataWidth()
               << "-bit beats, totalling " << bytes << " bytes";
    }

  return success();
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_burst_unwrap
static std::string pulpBurstUnwrapperSource(StringRef name,
                                            BurstUnwrapperOp unwrapper) {
  auto upstream = cast<PortType>(unwrapper.getUpstream().getType());
  // checkPulpBurstUnwrapperSupported has established one ID width.
  unsigned idWidth = upstream.getWriteIdWidth();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapper(os, name, upstream, "axi_burst_unwrap",
                       {"  input  logic clk_i", "  input  logic rst_ni"});

  os << "  axi_burst_unwrap #(\n";
  os << "    .MaxReadTxns  (" << std::max(upstream.getOutstandingReads(), 1u)
     << "),\n";
  os << "    .MaxWriteTxns (" << std::max(upstream.getOutstandingWrites(), 1u)
     << "),\n";
  os << "    .AddrWidth    (" << upstream.getAddrWidth() << "),\n";
  os << "    .DataWidth    (" << upstream.getDataWidth() << "),\n";
  os << "    .IdWidth      (" << idWidth << "),\n";
  os << "    .UserWidth    (" << pulpUserWidth(upstream) << "),\n";
  os << "    .axi_req_t    (" << prefix << "req_t),\n";
  os << "    .axi_resp_t   (" << prefix << "resp_t)\n";
  os << "  ) i_burst_unwrap (\n";
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

/// Report the demuxes PULP's axi_demux cannot express.
static LogicalResult checkPulpDemuxSupported(DemuxOp demux) {
  // Verifier already checked that upstream and downstream widths match
  return checkPulpIdWidths(demux, "axi_demux",
                           cast<PortType>(demux.getUpstream().getType()));
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_demux. PULP is
/// told which downstream port to route a request to rather than deriving it, so
/// the wrapper decodes the windows itself.
static std::string pulpDemuxSource(StringRef name, DemuxOp demux) {
  auto port = cast<PortType>(demux.getUpstream().getType());
  unsigned numDownstream = demux.getDownstream().size();
  unsigned addrWidth = port.getAddrWidth();
  // checkPulpDemuxSupported has established one ID width.
  unsigned idWidth = port.getWriteIdWidth();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitSymmetricWrapper(os, name, port, "axi_demux",
                       {"  input  logic clk_i", "  input  logic rst_ni"},
                       numDownstream);

  SmallVector<std::string> rules =
      pulpAddrMapRules(demux.getDownstream(), addrWidth);
  emitAddrMap(os, prefix, rules);

  // As wide as PULP's own select_t, so that both ends of the decode agree.
  unsigned selectWidth =
      numDownstream > 1 ? llvm::Log2_64_Ceil(numDownstream) : 1;
  os << "  typedef logic [" << selectWidth << "-1:0] select_t;\n";
  os << "  select_t aw_select, ar_select;\n";

  // PULP is handed the downstream port index of a request rather than deriving
  // it, so the address map is decoded here. An address no window covers goes
  // downstream to port 0.
  for (StringRef channel : {"aw", "ar"}) {
    os << "  addr_decode #(\n";
    os << "    .NoIndices        (" << numDownstream << "),\n";
    os << "    .NoRules          (" << rules.size() << "),\n";
    os << "    .addr_t           (" << prefix << "addr_t),\n";
    os << "    .rule_t           (rule_t)\n";
    os << "  ) i_" << channel << "_decode (\n";
    os << "    .addr_i           (slv_req." << channel << ".addr),\n";
    os << "    .addr_map_i       (AddrMap),\n";
    os << "    .idx_o            (" << channel << "_select),\n";
    os << "    .dec_valid_o      (),\n";
    os << "    .dec_error_o      (),\n";
    os << "    .en_default_idx_i (1'b1),\n";
    os << "    .default_idx_i    ('0)\n";
    os << "  );\n";
  }

  // The spill registers are left at the axi_demux defaults, since the op
  // carries no knobs for them.
  os << "  axi_demux #(\n";
  os << "    .AxiIdWidth  (" << idWidth << "),\n";
  os << "    .AtopSupport (1'b1),\n";
  os << pulpChannelParams(prefix) << ",\n";
  os << "    .NoMstPorts  (" << numDownstream << "),\n";
  os << "    .MaxTrans    ("
     << std::max(maxOutstanding(demux.getUpstream()), 1u) << "),\n";
  os << "    .AxiLookBits (" << idWidth << "),\n";
  os << "    .UniqueIds   (1'b0)\n";
  os << "  ) i_demux (\n";
  os << "    .clk_i           (clk_i),\n";
  os << "    .rst_ni          (rst_ni),\n";
  os << "    .test_i          (1'b0),\n";
  os << "    .slv_req_i       (slv_req),\n";
  os << "    .slv_aw_select_i (aw_select),\n";
  os << "    .slv_ar_select_i (ar_select),\n";
  os << "    .slv_resp_o      (slv_resp),\n";
  os << "    .mst_reqs_o      (mst_req),\n";
  os << "    .mst_resps_i     (mst_resp)\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

/// Report the muxes PULP's axi_mux cannot express.
static LogicalResult checkPulpMuxSupported(MuxOp mux) {
  // The verifier has established that the upstream ports all agree, so port #0
  // speaks for the upstream side.
  auto upstream = cast<PortType>(mux.getUpstream().front().getType());
  if (failed(checkPulpIdWidths(mux, "axi_mux", upstream, "upstream")))
    return failure();

  auto downstream = cast<PortType>(mux.getDownstream().getType());
  if (failed(checkPulpIdWidths(mux, "axi_mux", downstream, "downstream")))
    return failure();

  return checkPulpTagWidth(mux, "axi_mux", upstream, mux.getUpstream().size(),
                           downstream);
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_mux, with the
/// ports `mux`'s external module lowers to.
static std::string pulpMuxSource(StringRef name, MuxOp mux) {
  auto upstream = cast<PortType>(mux.getUpstream().front().getType());
  auto downstream = cast<PortType>(mux.getDownstream().getType());
  unsigned numUpstream = mux.getUpstream().size();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitDualIdWrapper(os, name, "axi_mux", upstream, numUpstream, downstream,
                    /*numDownstream=*/1);

  // The spill registers are left at the axi_mux defaults, since the op carries
  // no knobs for them. PULP drives one downstream port, so it takes the single
  // struct of the array the wrapper bridged to.
  os << "  axi_mux #(\n";
  os << "    .SlvAxiIDWidth (" << upstream.getWriteIdWidth() << "),\n";
  os << "    .slv_aw_chan_t (" << prefix << "slv_aw_chan_t),\n";
  os << "    .mst_aw_chan_t (" << prefix << "mst_aw_chan_t),\n";
  os << "    .w_chan_t      (" << prefix << "slv_w_chan_t),\n";
  os << "    .slv_b_chan_t  (" << prefix << "slv_b_chan_t),\n";
  os << "    .mst_b_chan_t  (" << prefix << "mst_b_chan_t),\n";
  os << "    .slv_ar_chan_t (" << prefix << "slv_ar_chan_t),\n";
  os << "    .mst_ar_chan_t (" << prefix << "mst_ar_chan_t),\n";
  os << "    .slv_r_chan_t  (" << prefix << "slv_r_chan_t),\n";
  os << "    .mst_r_chan_t  (" << prefix << "mst_r_chan_t),\n";
  os << "    .slv_req_t     (" << prefix << "slv_req_t),\n";
  os << "    .slv_resp_t    (" << prefix << "slv_resp_t),\n";
  os << "    .mst_req_t     (" << prefix << "mst_req_t),\n";
  os << "    .mst_resp_t    (" << prefix << "mst_resp_t),\n";
  os << "    .NoSlvPorts    (" << numUpstream << "),\n";
  os << "    .MaxWTrans     ("
     << std::max(maxOutstanding(mux.getUpstream()), 1u) << "),\n";
  os << "    .FallThrough   (1'b0)\n";
  os << "  ) i_mux (\n";
  os << "    .clk_i       (clk_i),\n";
  os << "    .rst_ni      (rst_ni),\n";
  os << "    .test_i      (1'b0),\n";
  os << "    .slv_reqs_i  (slv_req),\n";
  os << "    .slv_resps_o (slv_resp),\n";
  os << "    .mst_req_o   (mst_req[0]),\n";
  os << "    .mst_resp_i  (mst_resp[0])\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

/// Report the memory converters PULP's axi_to_mem cannot express.
static LogicalResult checkPulpToMemSupported(ToMemOp toMem) {
  auto port = cast<PortType>(toMem.getPort().getType());
  if (failed(checkPulpIdWidths(toMem, "axi_to_mem", port)))
    return failure();

  // PULP walks a burst up the memory's addresses, so only a single beat can
  // start anywhere else.
  for (WindowAttr window : port.getWindows().getWindows())
    for (BurstSpecAttr spec : window.getBurstSpecs().getBurstSpecs())
      if (spec.getKind() != BurstKind::Incr && spec.getLen() > 1)
        return toMem.emitOpError()
               << "cannot be lowered to a PULP axi_to_mem, which supports "
                  "bursts of more than one beat only where they increment, "
                  "because its port issues "
               << spec;

  return success();
}

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_to_mem. The
/// memory grants every request, so PULP's grant is tied high, and the atomics
/// it derives are left unconnected.
static std::string pulpToMemSource(StringRef name, ToMemOp toMem) {
  auto port = cast<PortType>(toMem.getPort().getType());
  // checkPulpToMemSupported has established one ID width.
  unsigned idWidth = port.getWriteIdWidth();
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  emitWrapperTypedefs(os, name, port, "axi_to_mem", {"mgr"});

  // The memory face carries the signals it already is, so only the upstream
  // manager's face explodes into channels.
  SmallVector<std::string> ports{"  input  logic clk_i",
                                 "  input  logic rst_ni",
                                 "  input  logic mem_rvalid_i",
                                 "  input  " + prefix + "data_t mem_rdata_i",
                                 "  output logic mem_req_o",
                                 "  output " + prefix + "addr_t mem_addr_o",
                                 "  output " + prefix + "data_t mem_wdata_o",
                                 "  output " + prefix + "strb_t mem_strb_o",
                                 "  output logic mem_we_o"};
  emitFacePorts(ports, prefix, "mgr", 0, /*isManager=*/false);
  os << "module " << name << " (\n" << llvm::join(ports, ",\n") << "\n);\n";

  os << "  " << prefix << "req_t  slv_req;\n";
  os << "  " << prefix << "resp_t slv_resp;\n";
  emitFaceBridge(os, port, "mgr", 0, /*isManager=*/false, "slv_req",
                 "slv_resp");

  // One bank as wide as the port, since the op drives a single memory. The
  // response buffer is left at the axi_to_mem default, since the op carries no
  // knob for the memory's latency.
  os << "  axi_to_mem #(\n";
  os << "    .axi_req_t  (" << prefix << "req_t),\n";
  os << "    .axi_resp_t (" << prefix << "resp_t),\n";
  os << "    .AddrWidth  (" << port.getAddrWidth() << "),\n";
  os << "    .DataWidth  (" << port.getDataWidth() << "),\n";
  os << "    .IdWidth    (" << idWidth << "),\n";
  os << "    .NumBanks   (1),\n";
  os << "    .BufDepth   (1)\n";
  os << "  ) i_to_mem (\n";
  os << "    .clk_i        (clk_i),\n";
  os << "    .rst_ni       (rst_ni),\n";
  os << "    .busy_o       (),\n";
  os << "    .axi_req_i    (slv_req),\n";
  os << "    .axi_resp_o   (slv_resp),\n";
  os << "    .mem_req_o    (mem_req_o),\n";
  os << "    .mem_gnt_i    (1'b1),\n";
  os << "    .mem_addr_o   (mem_addr_o),\n";
  os << "    .mem_wdata_o  (mem_wdata_o),\n";
  os << "    .mem_strb_o   (mem_strb_o),\n";
  os << "    .mem_atop_o   (),\n";
  os << "    .mem_we_o     (mem_we_o),\n";
  os << "    .mem_rvalid_i (mem_rvalid_i),\n";
  os << "    .mem_rdata_i  (mem_rdata_i)\n";
  os << "  );\n";
  os << "endmodule\n";
  return text;
}

LogicalResult circt::AXI4ToHW::checkPulpSupported(Operation *op) {
  if (failed(checkPulpIdsPresent(op)))
    return failure();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<XbarOp>(checkPulpXbarSupported)
      .Case<DemuxOp>(checkPulpDemuxSupported)
      .Case<MuxOp>(checkPulpMuxSupported)
      .Case<DWConverterOp>(checkPulpDWConverterSupported)
      .Case<IWConverterOp>(checkPulpIWConverterSupported)
      .Case<BurstSplitterOp>(checkPulpBurstSplitterSupported)
      .Case<BurstUnwrapperOp>(checkPulpBurstUnwrapperSupported)
      .Case<ToMemOp>(checkPulpToMemSupported)
      .Default(success());
}

void circt::AXI4ToHW::attachPulpSource(ImplicitLocOpBuilder &b,
                                       hw::HWModuleExternOp shape,
                                       Operation *op) {
  StringRef name = shape.getName();
  std::optional<std::string> text =
      TypeSwitch<Operation *, std::optional<std::string>>(op)
          .Case<XbarOp>([&](XbarOp xbar) { return pulpXbarSource(name, xbar); })
          .Case<DemuxOp>(
              [&](DemuxOp demux) { return pulpDemuxSource(name, demux); })
          .Case<MuxOp>([&](MuxOp mux) { return pulpMuxSource(name, mux); })
          .Case<CutOp>([&](CutOp cut) { return pulpCutSource(name, cut); })
          .Case<CDCOp>([&](CDCOp cdc) { return pulpCdcSource(name, cdc); })
          .Case<DWConverterOp>([&](DWConverterOp converter) {
            return pulpDWConverterSource(name, converter);
          })
          .Case<IWConverterOp>([&](IWConverterOp converter) {
            return pulpIWConverterSource(name, converter);
          })
          .Case<BurstSplitterOp>([&](BurstSplitterOp splitter) {
            return pulpBurstSplitterSource(name, splitter);
          })
          .Case<BurstUnwrapperOp>([&](BurstUnwrapperOp unwrapper) {
            return pulpBurstUnwrapperSource(name, unwrapper);
          })
          .Case<ToMemOp>(
              [&](ToMemOp toMem) { return pulpToMemSource(name, toMem); })
          .Default(std::nullopt);
  if (!text)
    return;

  auto source = sv::SVVerbatimSourceOp::create(
      b, b.getStringAttr(name + ".sv"), *text,
      hw::OutputFileAttr::getFromFilename(b.getContext(), name + ".sv"),
      b.getArrayAttr({}), /*additional_files=*/nullptr, b.getStringAttr(name));
  shape->setAttr("source", FlatSymbolRefAttr::get(source));
}
