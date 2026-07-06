//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4PASSES_H
#define CIRCT_DIALECT_AXI4_AXI4PASSES_H

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace circt {
namespace axi4 {

/// Generate the code for registering passes.
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/AXI4/AXI4Passes.h.inc"

} // namespace axi4
} // namespace circt

#endif // CIRCT_DIALECT_AXI4_AXI4PASSES_H
