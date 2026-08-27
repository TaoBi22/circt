//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Types.cpp.inc"

LogicalResult
axi4::verifyPortWidths(function_ref<InFlightDiagnostic()> emitError,
                       const Twine &prefix, uint32_t addrWidth,
                       uint32_t dataWidth) {
  if (addrWidth > 64)
    return emitError() << prefix << "'addr_width' must be at most 64, got "
                       << addrWidth;
  if (dataWidth < 8 || dataWidth > 1024 || !llvm::isPowerOf2_32(dataWidth))
    return emitError() << prefix
                       << "'data_width' must be a power of two between 8 "
                          "and 1024, got "
                       << dataWidth;
  return success();
}

LogicalResult PortType::verify(function_ref<InFlightDiagnostic()> emitError,
                               uint32_t addr_width, uint32_t data_width,
                               uint32_t write_id_width, uint32_t read_id_width,
                               uint32_t user_width, WindowSetAttr windows,
                               uint32_t outstanding_writes,
                               uint32_t outstanding_reads) {
  if (failed(verifyPortWidths(emitError, "port ", addr_width, data_width)))
    return failure();
  if (write_id_width > 32)
    return emitError() << "port 'write_id_width' must be at most 32, got "
                       << write_id_width;
  if (read_id_width > 32)
    return emitError() << "port 'read_id_width' must be at most 32, got "
                       << read_id_width;
  // Bounds computed in 64 bits to avoid 32-bit overflow if ID widths are 32.
  if (outstanding_writes > (uint64_t{1} << write_id_width))
    return emitError() << "port 'outstanding_writes' must be at most "
                       << (uint64_t{1} << write_id_width)
                       << " for a 'write_id_width' of " << write_id_width
                       << ", got " << outstanding_writes;
  if (outstanding_reads > (uint64_t{1} << read_id_width))
    return emitError() << "port 'outstanding_reads' must be at most "
                       << (uint64_t{1} << read_id_width)
                       << " for a 'read_id_width' of " << read_id_width
                       << ", got " << outstanding_reads;
  return success();
}

hw::StructType axi4::getChannelPayloadType(PortType port, AXI4Channel channel) {
  MLIRContext *ctx = port.getContext();
  auto field = [&](StringRef name, unsigned width) {
    return hw::StructType::FieldInfo{StringAttr::get(ctx, name),
                                     IntegerType::get(ctx, width)};
  };
  // Address channels differ only in which ID width they carry.
  auto addressFields = [&](unsigned idWidth) {
    return SmallVector<hw::StructType::FieldInfo>{
        field("id", idWidth),
        field("addr", port.getAddrWidth()),
        field("len", kLenWidth),
        field("size", kSizeWidth),
        field("burst", kBurstWidth),
        field("lock", kLockWidth),
        field("cache", kCacheWidth),
        field("prot", kProtWidth),
        field("qos", kQosWidth),
        field("region", kRegionWidth),
        field("user", port.getUserWidth())};
  };

  SmallVector<hw::StructType::FieldInfo> fields;
  switch (channel) {
  case AXI4Channel::AW:
    fields = addressFields(port.getWriteIdWidth());
    break;
  case AXI4Channel::AR:
    fields = addressFields(port.getReadIdWidth());
    break;
  case AXI4Channel::W:
    fields = {field("data", port.getDataWidth()),
              field("strb", port.getDataWidth() / 8), field("last", kLastWidth),
              field("user", port.getUserWidth())};
    break;
  case AXI4Channel::B:
    fields = {field("id", port.getWriteIdWidth()), field("resp", kRespWidth),
              field("user", port.getUserWidth())};
    break;
  case AXI4Channel::R:
    fields = {field("id", port.getReadIdWidth()),
              field("data", port.getDataWidth()), field("resp", kRespWidth),
              field("last", kLastWidth), field("user", port.getUserWidth())};
    break;
  }
  return hw::StructType::get(ctx, fields);
}

void AXI4Dialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "circt/Dialect/AXI4/AXI4Types.cpp.inc"
      >();
}
