//===- AXI4ToHW.h - AXI4 to HW conversion pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the pass which lowers the abstract AXI4 network dialect to
// the HW dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AXI4TOHW_H
#define CIRCT_CONVERSION_AXI4TOHW_H

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_AXI4TOHW
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createAXI4ToHWPass();

} // namespace circt

#endif // CIRCT_CONVERSION_AXI4TOHW_H
