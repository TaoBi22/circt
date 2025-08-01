//===- LegalizeMemrefs.cpp - handshake memref legalization pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the memref legalization pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace handshake {
#define GEN_PASS_DEF_HANDSHAKELEGALIZEMEMREFS
#include "circt/Dialect/Handshake/HandshakePasses.h.inc"
} // namespace handshake
} // namespace circt

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace {

struct HandshakeLegalizeMemrefsPass
    : public circt::handshake::impl::HandshakeLegalizeMemrefsBase<
          HandshakeLegalizeMemrefsPass> {
  void runOnOperation() override {
    func::FuncOp op = getOperation();
    if (op.isExternal())
      return;

    // Erase all memref.dealloc operations - this implies that we consider all
    // memref.alloc's in the IR to be "static", in the C sense. It is then up to
    // callers of the handshake module to determine whether a call to said
    // module implies a _call_ (shared semantics) or an _instance_.
    for (auto dealloc :
         llvm::make_early_inc_range(op.getOps<memref::DeallocOp>()))
      dealloc.erase();

    auto b = OpBuilder(op);

    // Convert any memref.copy to explicit store operations (scf loop in case of
    // an array).
    for (auto copy : llvm::make_early_inc_range(op.getOps<memref::CopyOp>())) {
      b.setInsertionPoint(copy);
      auto loc = copy.getLoc();
      auto src = copy.getSource();
      auto dst = copy.getTarget();
      auto memrefType = cast<MemRefType>(src.getType());
      if (!isUniDimensional(memrefType)) {
        llvm::errs() << "Cannot legalize multi-dimensional memref operation "
                     << copy
                     << ". Please run the memref flattening pass before this "
                        "pass.";
        signalPassFailure();
        return;
      }

      auto emitLoadStore = [&](Value index) {
        llvm::SmallVector<Value> indices = {index};
        auto loadValue = memref::LoadOp::create(b, loc, src, indices);
        memref::StoreOp::create(b, loc, loadValue, dst, indices);
      };

      auto n = memrefType.getShape()[0];

      if (n > 1) {
        auto lb = arith::ConstantIndexOp::create(b, loc, 0).getResult();
        auto ub = arith::ConstantIndexOp::create(b, loc, n).getResult();
        auto step = arith::ConstantIndexOp::create(b, loc, 1).getResult();

        scf::ForOp::create(
            b, loc, lb, ub, step, llvm::SmallVector<Value>(),
            [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
              emitLoadStore(iv);
              scf::YieldOp::create(b, loc);
            });
      } else
        emitLoadStore(arith::ConstantIndexOp::create(b, loc, 0));

      copy.erase();
    }
  };
};
} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeLegalizeMemrefsPass() {
  return std::make_unique<HandshakeLegalizeMemrefsPass>();
}
