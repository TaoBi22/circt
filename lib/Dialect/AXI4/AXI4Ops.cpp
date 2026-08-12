//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AXI4 ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Width helpers
//===----------------------------------------------------------------------===//

namespace {
/// A width field of a port type, named for diagnostics.
struct WidthField {
  llvm::StringLiteral name;
  uint32_t (PortType::*get)() const;
};
} // namespace

// The port width fields, the ones an xbar carries through unchanged first,
// followed by the ID widths it widens.
static constexpr WidthField kWidths[] = {
    {"addr_width", &PortType::getAddrWidth},
    {"data_width", &PortType::getDataWidth},
    {"user_width", &PortType::getUserWidth},
    {"write_id_width", &PortType::getWriteIdWidth},
    {"read_id_width", &PortType::getReadIdWidth}};
static constexpr size_t kNumSharedWidths = 3;

/// Verify that `port` agrees with `reference` on each of the widths in
/// `fields`.
static LogicalResult verifyWidthsMatch(Operation *op,
                                       ArrayRef<WidthField> fields,
                                       PortType port, const Twine &portDesc,
                                       PortType reference,
                                       const Twine &referenceDesc) {
  for (const WidthField &field : fields) {
    uint32_t width = (port.*field.get)();
    uint32_t expected = (reference.*field.get)();
    if (width != expected)
      return op->emitOpError()
             << portDesc << "'s '" << field.name << "' (" << width
             << ") must match " << referenceDesc << "'s (" << expected << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Window helpers
//===----------------------------------------------------------------------===//

/// Verify that a port's upstream and downstream windows cover the same
/// addresses, and that applying some mapping to the upstream window's burst
/// specs produces burst specs the downstream window supports
static LogicalResult verifyWindowsConvert(
    Operation *op, PortType upstream, PortType downstream,
    llvm::function_ref<FailureOr<BurstSpecAttr>(BurstSpecAttr)> convert,
    const Twine &description) {
  ArrayRef<WindowAttr> upWindows = upstream.getWindows().getWindows();
  ArrayRef<WindowAttr> downWindows = downstream.getWindows().getWindows();
  if (!llvm::equal(upWindows, downWindows, [](WindowAttr up, WindowAttr down) {
        return up.getBase() == down.getBase() && up.getLast() == down.getLast();
      }))
    return op->emitOpError("upstream and downstream windows must cover the "
                           "same addresses");

  for (auto [upWindow, downWindow] : llvm::zip_equal(upWindows, downWindows)) {
    SmallVector<BurstSpecAttr> converted;
    for (BurstSpecAttr spec : upWindow.getBurstSpecs().getBurstSpecs()) {
      FailureOr<BurstSpecAttr> mapped = convert(spec);
      if (failed(mapped))
        return failure();
      converted.push_back(*mapped);
    }

    auto expected = BurstSetAttr::get(op->getContext(), converted);
    if (!downWindow.getBurstSpecs().covers(expected))
      return op->emitOpError()
             << "downstream window must support at least " << expected << " ("
             << description << "), but supports " << downWindow.getBurstSpecs();
  }

  return success();
}

/// The downstream port and window covering `address`, or a null window if no
/// downstream port covers it.
static std::pair<size_t, WindowAttr> findDownstreamWindow(ValueRange downstream,
                                                          uint64_t address) {
  for (auto [i, value] : llvm::enumerate(downstream))
    for (WindowAttr window :
         cast<PortType>(value.getType()).getWindows().getWindows())
      if (window.getBase() <= address && address <= window.getLast())
        return {i, window};
  return {0, {}};
}

/// Verify that no two downstream ports' windows overlap, so routing is
/// unambiguous.
static LogicalResult verifyWindowsDisjoint(Operation *op,
                                           ValueRange downstream) {
  for (auto [i, value] : llvm::enumerate(downstream)) {
    auto windows = cast<PortType>(value.getType()).getWindows();
    for (auto [j, other] : llvm::enumerate(downstream.take_front(i)))
      if (windows.overlaps(cast<PortType>(other.getType()).getWindows()))
        return op->emitOpError() << "downstream ports #" << j << " and #" << i
                                 << " have overlapping windows";
  }
  return success();
}

/// Verify that every window `upstream` can access is routed downstream, to a
/// port supporting at least the bursts it issues there. Assumes the downstream
/// windows are disjoint, so the existence of a covering window is enough.
static LogicalResult verifyWindowsRouted(Operation *op, PortType upstream,
                                         const Twine &upstreamDesc,
                                         ValueRange downstream) {
  for (WindowAttr window : upstream.getWindows().getWindows()) {
    // Walk through addresses, skipping the ones we know are covered
    // Begin at the window's start
    for (uint64_t address = window.getBase();;) {
      // Make sure it's supported
      auto [j, covering] = findDownstreamWindow(downstream, address);
      if (!covering)
        return op->emitOpError()
               << "address 0x" << llvm::utohexstr(address, /*LowerCase=*/true)
               << ", in " << upstreamDesc
               << "'s windows, is not covered by any downstream port";
      if (!covering.getBurstSpecs().covers(window.getBurstSpecs()))
        return op->emitOpError()
               << "downstream port #" << j
               << " does not support all the bursts " << upstreamDesc
               << " issues at address 0x"
               << llvm::utohexstr(address, /*LowerCase=*/true)
               << "; upstream requires " << window.getBurstSpecs()
               << ", downstream supports " << covering.getBurstSpecs();
      // If we know that every remaining address in the upstream window is
      // covered by this downstream window, we're done
      if (covering.getLast() >= window.getLast())
        break;
      // Otherwise, skip ahead to the next address that we don't already know
      // is covered (the address directly after the end of the covering
      // downstream window)
      address = covering.getLast() + 1;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XbarOp
//===----------------------------------------------------------------------===//

LogicalResult XbarOp::verify() {
  ValueRange upstream = getUpstream();
  ValueRange downstream = getDownstream();
  if (upstream.empty())
    return emitOpError("must have at least one upstream port");
  if (downstream.empty())
    return emitOpError("must have at least one downstream port");

  // Make sure all upstream ports agree on widths
  auto upstreamTy = cast<PortType>(upstream.front().getType());
  for (auto [i, value] : llvm::enumerate(upstream.drop_front()))
    if (failed(verifyWidthsMatch(
            *this, kWidths, cast<PortType>(value.getType()),
            "upstream port #" + Twine(i + 1), upstreamTy, "upstream port #0")))
      return failure();

  // Each manager's transactions are tagged with its index downstream.
  uint32_t idBits = llvm::Log2_64_Ceil(upstream.size());

  // Make sure all downstream ports agree on address, data, and user widths
  for (auto [i, value] : llvm::enumerate(downstream)) {
    auto downstreamTy = cast<PortType>(value.getType());
    if (failed(verifyWidthsMatch(
            *this, ArrayRef(kWidths).take_front(kNumSharedWidths), downstreamTy,
            "downstream port #" + Twine(i), upstreamTy, "upstream port #0")))
      return failure();

    // Make sure downstream ports are wide enough to uniquely tag transactions
    // from upstream ports.
    for (const WidthField &field :
         ArrayRef(kWidths).drop_front(kNumSharedWidths)) {
      uint32_t least = (upstreamTy.*field.get)() + idBits;
      if ((downstreamTy.*field.get)() < least)
        return emitOpError()
               << "downstream port #" << i << "'s '" << field.name
               << "' must be at least " << least << " to tag transactions from "
               << upstream.size() << " managers, got "
               << (downstreamTy.*field.get)();
    }
  }

  if (failed(verifyWindowsDisjoint(*this, downstream)))
    return failure();

  for (auto [i, value] : llvm::enumerate(upstream))
    if (failed(verifyWindowsRouted(*this, cast<PortType>(value.getType()),
                                   "upstream port #" + Twine(i), downstream)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// DWConverterOp
//===----------------------------------------------------------------------===//

// The port width fields a data width converter leaves alone.
static constexpr WidthField kPreservedWidths[] = {
    {"addr_width", &PortType::getAddrWidth},
    {"user_width", &PortType::getUserWidth},
    {"write_id_width", &PortType::getWriteIdWidth},
    {"read_id_width", &PortType::getReadIdWidth}};

/// Calculates new burst length after a change in data width
static std::optional<uint32_t> convertLen(uint32_t len, uint32_t from,
                                          uint32_t to) {
  uint64_t bits = uint64_t{len} * from;
  if (bits % to != 0)
    return std::nullopt;
  return bits / to;
}

LogicalResult DWConverterOp::verify() {
  auto upstream = cast<PortType>(getUpstream().getType());
  auto downstream = cast<PortType>(getDownstream().getType());
  uint32_t width = downstream.getDataWidth();

  // A conversion changes the data width, and leaves every other width alone
  if (failed(verifyWidthsMatch(*this, kPreservedWidths, downstream,
                               "downstream port", upstream, "upstream port")))
    return failure();

  // A conversion re-widths, it does not re-address, and each upstream burst
  // becomes exactly one downstream burst of the same kind, carrying the same
  // bytes in beats of the new width
  return verifyWindowsConvert(
      *this, upstream, downstream,
      [&](BurstSpecAttr spec) -> FailureOr<BurstSpecAttr> {
        std::optional<uint32_t> len =
            convertLen(spec.getLen(), upstream.getDataWidth(), width);
        if (!len) {
          emitOpError() << "upstream burst " << spec
                        << " does not divide into whole " << width
                        << "-bit beats";
          return failure();
        }

        BurstSpecAttr beats = BurstSpecAttr::getChecked(
            [&] {
              return emitOpError() << "upstream burst " << spec << " has no "
                                   << width << "-bit equivalent: ";
            },
            getContext(), spec.getKind(), *len);
        if (!beats)
          return failure();
        return beats;
      },
      "the upstream's bursts in beats of " + Twine(width) + " bits");
}

//===----------------------------------------------------------------------===//
// BurstSplitterOp
//===----------------------------------------------------------------------===//

LogicalResult BurstSplitterOp::verify() {
  auto upstream = cast<PortType>(getUpstream().getType());
  auto downstream = cast<PortType>(getDownstream().getType());

  // A split changes burst lengths, and leaves every width alone
  if (failed(verifyWidthsMatch(*this, kWidths, downstream, "downstream port",
                               upstream, "upstream port")))
    return failure();

  // A split re-lengths, it does not re-address, and each upstream burst becomes
  // single beats of its own kind - except a `wrap`, whose wrapping belongs to
  // the sequence of beats rather than to any one of them
  return verifyWindowsConvert(
      *this, upstream, downstream,
      [&](BurstSpecAttr spec) -> FailureOr<BurstSpecAttr> {
        BurstKind kind = spec.getKind() == BurstKind::Wrap ? BurstKind::Incr
                                                           : spec.getKind();
        return BurstSpecAttr::get(getContext(), kind, /*len=*/1);
      },
      "the upstream's bursts split into single beats");
}

//===----------------------------------------------------------------------===//
// DemuxOp
//===----------------------------------------------------------------------===//

LogicalResult DemuxOp::verify() {
  auto upstream = cast<PortType>(getUpstream().getType());
  ValueRange downstream = getDownstream();
  if (downstream.empty())
    return emitOpError("must have at least one downstream port");

  // A demux routes, it does not re-width or re-tag
  for (auto [i, value] : llvm::enumerate(downstream))
    if (failed(verifyWidthsMatch(
            *this, kWidths, cast<PortType>(value.getType()),
            "downstream port #" + Twine(i), upstream, "upstream port")))
      return failure();

  if (failed(verifyWindowsDisjoint(*this, downstream)))
    return failure();

  return verifyWindowsRouted(*this, upstream, "upstream port", downstream);
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.cpp.inc"
