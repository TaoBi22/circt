//===- AXI4ToHWInternals.h - Shared AXI4 to HW lowering pieces ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pieces the AXI4 to HW lowering shares with its PULP mapping.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_AXI4TOHW_AXI4TOHWINTERNALS_H
#define CONVERSION_AXI4TOHW_AXI4TOHWINTERNALS_H

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace circt {
namespace AXI4ToHW {

/// An AXI4 channel's name, and whether a manager drives its payload.
struct ChannelInfo {
  axi4::AXI4Channel channel;
  llvm::StringLiteral name;
  bool isRequest;
};

/// The channels in AXI4 order, which every signal list follows.
inline constexpr ChannelInfo kChannels[] = {
    {axi4::AXI4Channel::AW, llvm::StringLiteral("aw"), true},
    {axi4::AXI4Channel::W, llvm::StringLiteral("w"), true},
    {axi4::AXI4Channel::B, llvm::StringLiteral("b"), false},
    {axi4::AXI4Channel::AR, llvm::StringLiteral("ar"), true},
    {axi4::AXI4Channel::R, llvm::StringLiteral("r"), false}};

/// Report the crossbars PULP's axi_xbar cannot express.
mlir::LogicalResult checkPulpSupported(axi4::XbarOp xbar);

/// Attach a PULP axi_xbar wrapper to `shape`, the external module `xbar` is
/// lowered to.
void attachPulpSource(mlir::ImplicitLocOpBuilder &b, hw::HWModuleExternOp shape,
                      axi4::XbarOp xbar);

} // namespace AXI4ToHW
} // namespace circt

#endif // CONVERSION_AXI4TOHW_AXI4TOHWINTERNALS_H
