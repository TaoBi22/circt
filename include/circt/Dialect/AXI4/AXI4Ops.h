//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_H
#define CIRCT_DIALECT_AXI4_AXI4OPS_H

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Interfaces.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

namespace circt {
namespace axi4 {
namespace OpTrait {
/// Constrains an op's results to at most one use each. Fan-out of an AXI4 port
/// must go through an `axi4.xbar`.
template <typename ConcreteType>
class ResultsAtMostOneUse
    : public mlir::OpTrait::TraitBase<ConcreteType, ResultsAtMostOneUse> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      if (result.hasNUsesOrMore(2))
        return op->emitOpError(
            "result must have at most one use; route through an "
            "'axi4.xbar' to fan out to multiple endpoints");
    return mlir::success();
  }
};
} // namespace OpTrait
} // namespace axi4
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.h.inc"

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_H
