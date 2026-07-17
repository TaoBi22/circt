//===- AXI4ToHWInternals.h - Shared AXI4-to-HW lowering internals ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Types and the NetworkLowering context for the AXI4-to-HW lowering, factored
// into a header so they can be shared across the pass's translation units.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CONVERSION_AXI4TOHW_AXI4TOHWINTERNALS_H
#define CONVERSION_AXI4TOHW_AXI4TOHWINTERNALS_H

#include "circt/Conversion/AXI4ToHW.h"
#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <array>
#include <optional>

namespace circt {
namespace AXI4ToHW {

//===----------------------------------------------------------------------===//
// Canonical network signal representation
//===----------------------------------------------------------------------===//

/// One network signal that may or may not be a backedge.
struct Wire {
  mlir::Value value;
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

//===----------------------------------------------------------------------===//
// Port group / channel metadata
//===----------------------------------------------------------------------===//

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
  mlir::Operation *defOp;
  bool isManager;
  axi4::PortType portType;
  MappingKind kind;
  // Prefix for port names according to mapping
  std::string name;
  mlir::Value axiClock;
  std::string clockPort;
  mlir::Value axiReset;
  std::string resetPort;
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
  llvm::StringRef token;
  bool isRequest;
};
extern const ChannelInfo kChannelInfos[kNumChannels];

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Build the `hw.struct` payload type for one channel of an `!axi4.port`.
hw::StructType getChannelPayloadType(axi4::PortType port, AXI4Channel channel);

/// Append channel-split ports (payload, valid, ready) for a single !axi4.port.
void buildChannelPortList(mlir::MLIRContext *ctx, axi4::PortType portType,
                          bool isManager, llvm::StringRef prefix,
                          llvm::SmallVectorImpl<hw::PortInfo> &ports);

/// Get the exclusive end of the window - returns nullopt if it does not fit the
/// `addrW`-bit address space.
std::optional<uint64_t> windowEnd(uint64_t base, uint64_t size, unsigned addrW);

/// Reject crossbars the PULP axi_xbar backend cannot represent.
mlir::LogicalResult checkXbarSupported(axi4::XbarOp xbar);

/// Reject data width converters the PULP axi_dw_converter backend cannot
/// represent (it only resizes data; ID/address/user must be preserved and it
/// uses a single ID width).
mlir::LogicalResult checkDwConverterSupported(axi4::DWConverterOp dwc);

/// One `xbar_rule_*_t` entry: requests in [start, end) route to master port
/// `idx`. Following PULP's addr_decode, `end == 0` means "to the end of the
/// address space".
struct AddrRule {
  unsigned idx;
  uint64_t start;
  uint64_t end;
};

//===----------------------------------------------------------------------===//
// NetworkLowering context
//
// Orchestrates lowering the whole AXI4 network.
//===----------------------------------------------------------------------===//

class NetworkLowering {
public:
  NetworkLowering(mlir::ModuleOp module)
      : module(module), builder(module.getLoc(), module.getContext()),
        bb(builder, module.getLoc()) {
    builder.setInsertionPointToEnd(module.getBody());
  }

  mlir::LogicalResult run();

private:
  mlir::LogicalResult lowerNetwork();
  mlir::LogicalResult lowerNode(axi4::NodeOp node);
  mlir::LogicalResult lowerXbar(axi4::XbarOp xbar);
  mlir::LogicalResult lowerCut(axi4::CutOp cut);
  mlir::LogicalResult lowerCdc(axi4::CDCOp cdc);
  mlir::LogicalResult lowerDwConverter(axi4::DWConverterOp dwc);
  /// Get (or create) the PULP `axi_xbar` wrapper for this crossbar shape and
  /// address map, deduplicated by their combined signature.
  sv::SVVerbatimModuleOp getOrCreateXbarModule(unsigned numUpstream,
                                               unsigned numDownstream,
                                               axi4::PortType upstreamType,
                                               axi4::PortType downstreamType,
                                               llvm::ArrayRef<AddrRule> rules);
  /// Get (or create) the PULP `axi_cut` wrapper for this port shape,
  /// deduplicated by shape.
  sv::SVVerbatimModuleOp getOrCreateCutModule(axi4::PortType pt);
  /// Get (or create) the PULP `axi_cdc` wrapper for this port shape,
  /// deduplicated by shape.
  sv::SVVerbatimModuleOp getOrCreateCdcModule(axi4::PortType pt);
  /// Get (or create) the PULP `axi_dw_converter` wrapper for this
  /// upstream/downstream port shape, deduplicated by shape.
  sv::SVVerbatimModuleOp getOrCreateDwConverterModule(axi4::PortType upType,
                                                      axi4::PortType downType);
  /// Instantiate `moduleOp`, wiring each interface in `specs`; returns the
  /// per-interface wires in `wiresOut`.
  mlir::LogicalResult buildInstance(mlir::Operation *diag,
                                    hw::HWModuleLike moduleOp,
                                    llvm::StringRef instanceName,
                                    llvm::ArrayRef<PortGroupSpec> specs,
                                    llvm::SmallVectorImpl<PortWires> &wiresOut);
  /// Materialize an `!axi4.clock` value as the module clock port's type.
  mlir::Value materializeClock(mlir::Value axiClock, mlir::Type portType);
  /// Materialize an `!axi4.reset` value as the module reset port's type.
  mlir::Value materializeReset(mlir::Value axiReset, mlir::Type portType);

  mlir::ModuleOp module;
  mlir::ImplicitLocOpBuilder builder;
  BackedgeBuilder bb;
  llvm::DenseMap<mlir::OpOperand *, EdgeWires> edges;
  llvm::DenseMap<mlir::Value, mlir::Value> clockCache;
  llvm::DenseMap<mlir::Value, mlir::Value> resetCache;
  /// Emitted xbar wrappers, keyed by shape+address-map signature.
  llvm::StringMap<sv::SVVerbatimModuleOp> xbarWrappers;
  /// Emitted cut wrappers, keyed by port shape.
  llvm::StringMap<sv::SVVerbatimModuleOp> cutWrappers;
  /// Emitted cdc wrappers, keyed by port shape.
  llvm::StringMap<sv::SVVerbatimModuleOp> cdcWrappers;
  /// Emitted data-width-converter wrappers, keyed by up/down port shape.
  llvm::StringMap<sv::SVVerbatimModuleOp> dwConverterWrappers;
  unsigned instanceCounter = 0;
};

} // namespace AXI4ToHW
} // namespace circt

#endif // CONVERSION_AXI4TOHW_AXI4TOHWINTERNALS_H
