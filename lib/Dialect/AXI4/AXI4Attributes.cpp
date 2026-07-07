//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

#include "circt/Dialect/AXI4/AXI4Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Attributes.cpp.inc"

LogicalResult
BurstSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      BurstKind kind, std::optional<uint32_t> len) {
  // A 'len' is present iff the burst is incrementing ('incr' or 'wrap').
  if (kind == BurstKind::Fixed && len.has_value())
    return emitError() << "'fixed' burst kind cannot have a 'len'";
  if (kind != BurstKind::Fixed && !len.has_value())
    return emitError() << "'" << stringifyBurstKind(kind)
                       << "' burst kind requires a 'len'";
  return success();
}

StringRef PortWiresAttr::getClockPort() const { return getClock(); }
StringRef ReqRespStructsAttr::getClockPort() const { return getClock(); }
StringRef PortInterfaceAttr::getClockPort() const { return getClock(); }

void AXI4Dialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/AXI4/AXI4Attributes.cpp.inc"
      >();
}
