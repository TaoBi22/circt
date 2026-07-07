//===- AXI4ToHW.cpp - Translate AXI4 into HW -------------------------------===//
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
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_AXI4TOHW
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::axi4;

//===----------------------------------------------------------------------===//
// Canonical channel representation
//===----------------------------------------------------------------------===//

namespace {

/// One AXI4 channel's three RTL signals
struct ChannelWires {
  Value payload;
  Value valid;
  Value ready;
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

/// The five AXI4 channels, in canonical order.
enum class AXI4Channel { AW, W, B, AR, R };
constexpr unsigned kNumChannels = 5;

// TODO: drop [[maybe_unused]] once the synthesis patterns use these.

/// Build the `hw.struct` payload type for one channel of an `!axi4.port`.
[[maybe_unused]] hw::StructType getChannelPayloadType(PortType port,
                                                      AXI4Channel channel) {
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
              field("user", port.getUserReqWidth())};
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
              field("user", port.getUserReqWidth())};
    break;
  case AXI4Channel::W:
    fields = {field("data", port.getDataWidth()),
              field("strb", port.getDataWidth() / 8),
              field("last", kLastWidth),
              field("user", port.getUserDataWidth())};
    break;
  case AXI4Channel::B:
    fields = {field("id", port.getWriteIdWidth()), field("resp", kRespWidth),
              field("user", port.getUserRespWidth())};
    break;
  case AXI4Channel::R:
    // The R channel carries both data and response, so its user signal is the
    // concatenation of the data and response user widths (data bits high,
    // resp bits low), per the AXI4 spec's user-signaling convention.
    fields = {field("id", port.getReadIdWidth()),
              field("data", port.getDataWidth()), field("resp", kRespWidth),
              field("last", kLastWidth),
              field("user", port.getUserDataWidth() + port.getUserRespWidth())};
    break;
  }
  return hw::StructType::get(ctx, fields);
}

/// `!axi4.port` -> per-channel signals (payload, valid, ready x5);
/// `!axi4.clock` -> `!seq.clock`; `!axi4.reset` -> `i1`.
[[maybe_unused]] void populateAXI4TypeConversions(TypeConverter &converter) {
  converter.addConversion([](ClockType clock) -> Type {
    return seq::ClockType::get(clock.getContext());
  });
  converter.addConversion([](ResetType reset) -> Type {
    return IntegerType::get(reset.getContext(), 1);
  });
  converter.addConversion(
      [](PortType port, SmallVectorImpl<Type> &results) -> LogicalResult {
        Type i1 = IntegerType::get(port.getContext(), 1);
        for (unsigned i = 0; i < kNumChannels; ++i) {
          results.push_back(
              getChannelPayloadType(port, static_cast<AXI4Channel>(i)));
          results.push_back(i1); // valid
          results.push_back(i1); // ready
        }
        return success();
      });
}

struct AXI4ToHWPass : public circt::impl::AXI4ToHWBase<AXI4ToHWPass> {
  void runOnOperation() override;
};

} // namespace

void AXI4ToHWPass::runOnOperation() {
  // TODO: type conversion, network synthesis, and the port_wires adapter.
  getOperation().emitError("lower-axi4-to-hw: pass not yet implemented");
  signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::createAXI4ToHWPass() {
  return std::make_unique<AXI4ToHWPass>();
}
