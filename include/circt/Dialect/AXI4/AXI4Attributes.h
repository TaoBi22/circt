//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_H
#define CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_H

#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/BuiltinAttributes.h"

// Pull in the enum definitions.
#include "circt/Dialect/AXI4/AXI4Enums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Attributes.h.inc"

namespace circt {
namespace axi4 {

/// The longest burst of a kind AXI4 permits.
uint32_t getMaxBurstLen(BurstKind kind);

} // namespace axi4
} // namespace circt

#endif // CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_H
