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

/// Report the crossbars PULP's axi_xbar cannot express.
static LogicalResult checkPulpXbarSupported(XbarOp xbar) {
  // PULP has additional restrictions on ID widths (it uses a single width for
  // reads and writes, and all downstream ports must have the same width)
  auto checkIdWidths = [&](PortType port, const Twine &side) -> LogicalResult {
    if (port.getWriteIdWidth() == port.getReadIdWidth())
      return success();
    return xbar.emitOpError()
           << "cannot be lowered to a PULP axi_xbar, which uses a single ID "
              "width per side, because its "
           << side << " write ID width (" << port.getWriteIdWidth()
           << ") and read ID width (" << port.getReadIdWidth() << ") differ";
  };

  auto upstream = cast<PortType>(xbar.getUpstream().front().getType());
  if (failed(checkIdWidths(upstream, "upstream")))
    return failure();

  auto downstream = cast<PortType>(xbar.getDownstream().front().getType());
  if (failed(checkIdWidths(downstream, "downstream")))
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

  // The op verifier only asks the downstream ports to be wide enough to tag the
  // managers, but PULP derives its downstream width as exactly that and asserts
  // on the channel types it is handed, so a wider one fails at elaboration.
  uint32_t idBits = llvm::Log2_64_Ceil(xbar.getUpstream().size());
  if (downstream.getWriteIdWidth() != upstream.getWriteIdWidth() + idBits)
    return xbar.emitOpError()
           << "cannot be lowered to a PULP axi_xbar, which widens IDs by "
              "exactly the "
           << idBits << " bits needed to tag " << xbar.getUpstream().size()
           << (xbar.getUpstream().size() == 1 ? " manager" : " managers")
           << ", so its downstream ID width must be "
           << upstream.getWriteIdWidth() + idBits << ", not "
           << downstream.getWriteIdWidth();

  return success();
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

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_xbar, with the
/// ports `xbar`'s external module lowers to.
static std::string pulpXbarSource(StringRef name, XbarOp xbar) {
  auto upstream = cast<PortType>(xbar.getUpstream().front().getType());
  auto downstream = cast<PortType>(xbar.getDownstream().front().getType());
  unsigned numUpstream = xbar.getUpstream().size();
  unsigned numDownstream = xbar.getDownstream().size();
  unsigned addrWidth = upstream.getAddrWidth();
  unsigned dataWidth = upstream.getDataWidth();
  // checkPulpSupported has established one ID width per side.
  unsigned upstreamId = upstream.getWriteIdWidth();
  unsigned downstreamId = downstream.getWriteIdWidth();
  unsigned userWidth = pulpUserWidth(upstream);
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// Generated by --lower-axi4-to-hw=pulp-mapping: a wrapper around the "
        "PULP Platform axi_xbar.\n";
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
  // The crossbar routes over one ID width per side.
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

  // The crossbar is the subordinate to each upstream manager, and the manager
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

  // One rule per window of each downstream port. The AXI dialect has an
  // inclusive last address, and PULP's end_addr is exclusive. PULP uses an end
  // address of 0 to indicate wrapping to the end of the address space -
  // conveniently if we get the exclusive last address of an AXI window by
  // adding 1, we also wrap round to 0.
  uint64_t mask =
      addrWidth == 64 ? ~uint64_t{0} : (uint64_t{1} << addrWidth) - 1;
  SmallVector<std::string> rules;
  for (auto [index, value] : llvm::enumerate(xbar.getDownstream())) {
    for (WindowAttr window :
         cast<PortType>(value.getType()).getWindows().getWindows())
      rules.push_back(("    '{idx: " + Twine(index) +
                       ", start_addr: " + Twine(addrWidth) + "'h" +
                       llvm::utohexstr(window.getBase()) +
                       ", end_addr: " + Twine(addrWidth) + "'h" +
                       llvm::utohexstr((window.getLast() + 1) & mask) + "}")
                          .str());
  }

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
  os << "    AxiDataWidth:       " << dataWidth << ",\n";
  os << "    NoAddrRules:        " << rules.size() << ",\n";
  os << "    default:            '0\n";
  os << "  };\n";
  // A rule's addresses have to be as wide as the ones the crossbar decodes, so
  // the map carries its own struct rather than an `axi_pkg` one, whose
  // addresses are always 32 or 64 bits wide.
  os << "  typedef struct packed { int unsigned idx; " << prefix
     << "addr_t start_addr; " << prefix << "addr_t end_addr; } rule_t;\n";
  os << "  localparam rule_t [" << rules.size() << "-1:0] AddrMap = '{\n";
  os << llvm::join(rules, ",\n") << "\n";
  os << "  };\n";

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

/// A SystemVerilog wrapper named `name` instantiating PULP's axi_cut, with the
/// ports `cut`'s external module lowers to. Both faces carry the same port, and
/// the write and read ID widths stay independent, since axi_cut never inspects
/// them.
static std::string pulpCutSource(StringRef name, CutOp cut) {
  auto port = cast<PortType>(cut.getUpstream().getType());
  std::string prefix = (name + "_").str();

  std::string text;
  llvm::raw_string_ostream os(text);
  os << "// Generated by --lower-axi4-to-hw=pulp-mapping: a wrapper around the "
        "PULP Platform axi_cut.\n";
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
  for (const ChannelInfo &info : kChannels)
    emitPayloadStruct(os, prefix, "mgr", port, info, fieldType(info));
  for (const ChannelInfo &info : kChannels)
    emitPayloadStruct(os, prefix, "sub", port, info, fieldType(info));
  os << "\n";

  // The cut is the subordinate to its upstream manager, and the manager to its
  // downstream subordinate.
  SmallVector<std::string> ports{"  input  logic clk_i",
                                 "  input  logic rst_ni"};
  emitFacePorts(ports, prefix, "mgr", 0, /*isManager=*/false);
  emitFacePorts(ports, prefix, "sub", 0, /*isManager=*/true);
  os << "module " << name << " (\n" << llvm::join(ports, ",\n") << "\n);\n";

  os << "  " << prefix << "req_t  slv_req;\n";
  os << "  " << prefix << "resp_t slv_resp;\n";
  os << "  " << prefix << "req_t  mst_req;\n";
  os << "  " << prefix << "resp_t mst_resp;\n";

  emitFaceBridge(os, port, "mgr", 0, /*isManager=*/false, "slv_req",
                 "slv_resp");
  emitFaceBridge(os, port, "sub", 0, /*isManager=*/true, "mst_req", "mst_resp");

  os << "  axi_cut #(\n";
  os << "    .Bypass     (1'b0),\n";
  os << "    .aw_chan_t  (" << prefix << "aw_chan_t),\n";
  os << "    .w_chan_t   (" << prefix << "w_chan_t),\n";
  os << "    .b_chan_t   (" << prefix << "b_chan_t),\n";
  os << "    .ar_chan_t  (" << prefix << "ar_chan_t),\n";
  os << "    .r_chan_t   (" << prefix << "r_chan_t),\n";
  os << "    .axi_req_t  (" << prefix << "req_t),\n";
  os << "    .axi_resp_t (" << prefix << "resp_t)\n";
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

LogicalResult circt::AXI4ToHW::checkPulpSupported(Operation *op) {
  if (failed(checkPulpIdsPresent(op)))
    return failure();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<XbarOp>(checkPulpXbarSupported)
      .Default(success());
}

void circt::AXI4ToHW::attachPulpSource(ImplicitLocOpBuilder &b,
                                       hw::HWModuleExternOp shape,
                                       Operation *op) {
  StringRef name = shape.getName();
  std::optional<std::string> text =
      TypeSwitch<Operation *, std::optional<std::string>>(op)
          .Case<XbarOp>([&](XbarOp xbar) { return pulpXbarSource(name, xbar); })
          .Case<CutOp>([&](CutOp cut) { return pulpCutSource(name, cut); })
          .Default(std::nullopt);
  if (!text)
    return;

  auto source = sv::SVVerbatimSourceOp::create(
      b, b.getStringAttr(name + ".sv"), *text,
      hw::OutputFileAttr::getFromFilename(b.getContext(), name + ".sv"),
      b.getArrayAttr({}), /*additional_files=*/nullptr, b.getStringAttr(name));
  shape->setAttr("source", FlatSymbolRefAttr::get(source));
}
