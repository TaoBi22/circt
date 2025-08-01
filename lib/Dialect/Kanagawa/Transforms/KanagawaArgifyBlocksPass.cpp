//===- KanagawaArgifyBlocksPass.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWAARGIFYBLOCKS
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;

namespace {

// A mapping between values and op operands who use them. A MapVector is used to
// ensure determinism.
using ValueMapping = llvm::MapVector<Value, llvm::SmallVector<OpOperand *>>;

// Returns a map of values to op operands, of values that are defined
// outside of the block op.
static void getExternallyDefinedOperands(StaticBlockOp blockOp,
                                         ValueMapping &mapping) {
  Block *blockBodyBlock = blockOp.getBodyBlock();
  for (Operation &op : *blockBodyBlock) {
    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      if (v.getParentBlock() != blockBodyBlock)
        mapping[v].push_back(&operand);
    }
  }
}

struct BlockConversionPattern : public OpConversionPattern<StaticBlockOp> {
  using OpConversionPattern<StaticBlockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StaticBlockOp blockOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueMapping mapping;
    getExternallyDefinedOperands(blockOp, mapping);
    Block *bodyBlock = blockOp.getBodyBlock();

    auto isolatedBlock = IsolatedStaticBlockOp::create(
        rewriter, blockOp.getLoc(), blockOp.getResultTypes(),
        blockOp.getOperands(), blockOp.getMaxThreadsAttr());
    // Erase the default terminator.
    Block *isolatedBlockBody = isolatedBlock.getBodyBlock();
    rewriter.eraseOp(isolatedBlockBody->getTerminator());
    llvm::SmallVector<Value> preAddBArgs;
    llvm::copy(blockOp.getBodyBlock()->getArguments(),
               std::back_inserter(preAddBArgs));

    // Add inputs and block arguments to the block, and replace the operand
    // uses.
    for (auto &[value, uses] : mapping) {
      isolatedBlock.getInputsMutable().append({value});
      auto newArg =
          isolatedBlockBody->addArgument(value.getType(), value.getLoc());
      for (OpOperand *operand : uses)
        operand->set(newArg);
    }
    // Inline the old block into the isolated block
    rewriter.mergeBlocks(bodyBlock, isolatedBlockBody, preAddBArgs);
    rewriter.replaceOp(blockOp, isolatedBlock.getResults());
    return success();
  }
};

struct ArgifyBlocksPass
    : public circt::kanagawa::impl::KanagawaArgifyBlocksBase<ArgifyBlocksPass> {
  void runOnOperation() override;
};
} // anonymous namespace

void ArgifyBlocksPass::runOnOperation() {
  auto *ctx = &getContext();
  ConversionTarget target(*ctx);
  target.addIllegalOp<StaticBlockOp>();
  target.addLegalDialect<KanagawaDialect>();

  RewritePatternSet patterns(ctx);
  patterns.add<BlockConversionPattern>(ctx);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::kanagawa::createArgifyBlocksPass() {
  return std::make_unique<ArgifyBlocksPass>();
}
