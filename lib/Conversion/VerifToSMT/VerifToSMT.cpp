//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a verif::AssertOp operation with an i1 operand to a smt::AssertOp.
struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    return success();
  }
};

/// Lower a verif::LecOp operation to a miter circuit encoded in SMT.
/// More information on miter circuits can be found, e.g., in this paper:
/// Brand, D., 1993, November. Verification of large synthesized designs. In
/// Proceedings of 1993 International Conference on Computer Aided Design
/// (ICCAD) (pp. 534-537). IEEE.
struct LogicEquivalenceCheckingOpConversion
    : OpConversionPattern<verif::LogicEquivalenceCheckingOp> {
  using OpConversionPattern<
      verif::LogicEquivalenceCheckingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      Value trueVal =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    smt::SolverOp solver =
        rewriter.create<smt::SolverOp>(loc, rewriter.getI1Type(), ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(rewriter.create<smt::DeclareFunOp>(loc, arg.getType()));

    // Third, inline the blocks
    // Note: the argument value replacement does not happen immediately, but
    // only after all the operations are already legalized.
    // Also, it has to be ensured that the original argument type and the type
    // of the value with which is is to be replaced match. The value is looked
    // up (transitively) in the replacement map at the time the replacement
    // pattern is committed.
    rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Fourth, convert the yielded values back to the source type system (since
    // the operations of the inlined blocks will be converted by other patterns
    // later on and we should make sure the IR is well-typed after each pattern
    // application), and build the 'assert'.
    SmallVector<Value> outputsDifferent;
    for (auto [out1, out2] :
         llvm::zip(firstOutputs->getOperands(), secondOutputs->getOperands())) {
      Value o1 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out1);
      Value o2 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out2);
      outputsDifferent.emplace_back(
          rewriter.create<smt::DistinctOp>(loc, o1, o2));
    }

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    Value toAssert;
    if (outputsDifferent.size() == 1)
      toAssert = outputsDifferent[0];
    else
      toAssert = rewriter.create<smt::OrOp>(loc, outputsDifferent);

    rewriter.create<smt::AssertOp>(loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    Value falseVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value trueVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    auto checkOp = rewriter.create<smt::CheckOp>(loc, rewriter.getI1Type());
    rewriter.createBlock(&checkOp.getSatRegion());
    rewriter.create<smt::YieldOp>(loc, falseVal);
    rewriter.createBlock(&checkOp.getUnknownRegion());
    rewriter.create<smt::YieldOp>(loc, falseVal);
    rewriter.createBlock(&checkOp.getUnsatRegion());
    rewriter.create<smt::YieldOp>(loc, trueVal);
    rewriter.setInsertionPointAfter(checkOp);
    rewriter.create<smt::YieldOp>(loc, checkOp->getResults());

    rewriter.replaceOp(op, solver->getResults());
    return success();
  }
};

///
struct VerifBMCOpConversion : OpConversionPattern<verif::BMCOp> {
  using OpConversionPattern<verif::BMCOp>::OpConversionPattern;

  VerifBMCOpConversion(TypeConverter &converter, MLIRContext *context,
                       Namespace &names)
      : OpConversionPattern(converter, context), names(names) {}

  LogicalResult
  matchAndRewrite(verif::BMCOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    SmallVector<Type> oldInputTy(op.getCircuit().getArgumentTypes());
    SmallVector<Type> inputTy, outputTy;
    if (failed(typeConverter->convertTypes(oldInputTy, inputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), outputTy)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getCircuit(), *typeConverter)))
      return failure();

    unsigned numRegs =
        cast<IntegerAttr>(op->getAttr("num_regs")).getValue().getZExtValue();

    auto funcTy = rewriter.getFunctionType(inputTy, outputTy);
    func::FuncOp funcOp;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          op->getParentOfType<ModuleOp>().getBody());
      funcOp = rewriter.create<func::FuncOp>(loc, names.newName("bmc"), funcTy);
      rewriter.inlineRegionBefore(op.getCircuit(), funcOp.getFunctionBody(),
                                  funcOp.end());
      auto operands = funcOp.getBody().front().back().getOperands();
      rewriter.eraseOp(&funcOp.getFunctionBody().front().back());
      rewriter.setInsertionPointToEnd(&funcOp.getBody().front());
      SmallVector<Value> toReturn;
      for (unsigned i = 0; i < outputTy.size(); ++i)
        toReturn.push_back(typeConverter->materializeTargetConversion(
            rewriter, loc, outputTy[i], operands[i]));
      rewriter.create<func::ReturnOp>(loc, toReturn);
    }

    auto solver =
        rewriter.create<smt::SolverOp>(loc, rewriter.getI1Type(), ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    SmallVector<Value> inputDecls;
    for (auto [oldTy, newTy] : llvm::zip(oldInputTy, inputTy)) {
      if (isa<seq::ClockType>(oldTy))
        inputDecls.push_back(rewriter.create<smt::BVConstantOp>(
            loc, smt::BitVectorAttr::get(getContext(), 0, 1)));
      else
        inputDecls.push_back(rewriter.create<smt::DeclareFunOp>(loc, newTy));
    }

    Value lowerBound =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
    Value step =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(1));
    Value upperBound =
        rewriter.create<arith::ConstantOp>(loc, adaptor.getBoundAttr());
    Value constFalse =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value constTrue =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    inputDecls.push_back(constFalse); // wasViolated?
    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, inputDecls,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          ValueRange phaseOneOuts =
              builder.create<func::CallOp>(loc, funcOp, iterArgs.drop_back())
                  ->getResults();
          auto checkOp =
              rewriter.create<smt::CheckOp>(loc, builder.getI1Type());
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&checkOp.getSatRegion());
            builder.create<smt::YieldOp>(loc, constTrue);
            builder.createBlock(&checkOp.getUnknownRegion());
            builder.create<smt::YieldOp>(loc, constTrue);
            builder.createBlock(&checkOp.getUnsatRegion());
            builder.create<smt::YieldOp>(loc, constFalse);
          }

          Value violated = builder.create<arith::OrIOp>(
              loc, checkOp.getResult(0), iterArgs.back());

          SmallVector<Value> newDecls;
          for (auto [oldTy, newTy] :
               llvm::zip(TypeRange(oldInputTy).drop_back(numRegs),
                         TypeRange(inputTy).drop_back(numRegs))) {
            if (isa<seq::ClockType>(oldTy))
              newDecls.push_back(builder.create<smt::BVConstantOp>(
                  loc, smt::BitVectorAttr::get(getContext(), 1, 1)));
            else
              newDecls.push_back(builder.create<smt::DeclareFunOp>(loc, newTy));
          }

          newDecls.append(llvm::to_vector(phaseOneOuts.take_back(numRegs)));

          ValueRange phaseTwoOuts =
              builder.create<func::CallOp>(loc, funcOp, newDecls)->getResults();
          auto phaseTwoCheckOp =
              builder.create<smt::CheckOp>(loc, builder.getI1Type());
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&phaseTwoCheckOp.getSatRegion());
            builder.create<smt::YieldOp>(loc, constTrue);
            builder.createBlock(&phaseTwoCheckOp.getUnknownRegion());
            builder.create<smt::YieldOp>(loc, constTrue);
            builder.createBlock(&phaseTwoCheckOp.getUnsatRegion());
            builder.create<smt::YieldOp>(loc, constFalse);
          }
          violated = builder.create<arith::OrIOp>(
              loc, phaseTwoCheckOp.getResult(0), violated);

          newDecls.clear();
          for (auto [oldTy, newTy] :
               llvm::zip(TypeRange(oldInputTy).drop_back(numRegs),
                         TypeRange(inputTy).drop_back(numRegs))) {
            if (isa<seq::ClockType>(oldTy))
              newDecls.push_back(builder.create<smt::BVConstantOp>(
                  loc, smt::BitVectorAttr::get(getContext(), 0, 1)));
            else
              newDecls.push_back(builder.create<smt::DeclareFunOp>(loc, newTy));
          }

          newDecls.append(SmallVector<Value>(phaseTwoOuts.take_back(numRegs)));
          newDecls.push_back(violated);

          builder.create<scf::YieldOp>(loc, newDecls);
        });

    Value res = rewriter.create<arith::XOrIOp>(loc, forOp->getResults().back(),
                                               constTrue);
    rewriter.create<smt::YieldOp>(loc, res);

    rewriter.replaceOp(op, solver.getResults());

    return success();
  }

  Namespace &names;
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Verif to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertVerifToSMTPass
    : public circt::impl::ConvertVerifToSMTBase<ConvertVerifToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateVerifToSMTConversionPatterns(TypeConverter &converter,
                                                 RewritePatternSet &patterns,
                                                 Namespace &names) {
  patterns.add<VerifAssertOpConversion, LogicEquivalenceCheckingOpConversion>(
      converter, patterns.getContext());
  patterns.add<VerifBMCOpConversion>(converter, patterns.getContext(), names);
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<verif::VerifDialect>();
  target.addLegalDialect<smt::SMTDialect, arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);

  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  Namespace names;
  names.add(symCache);
  populateVerifToSMTConversionPatterns(converter, patterns, names);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // Cleanup as many unrealized conversion casts as possible. This is applied
  // separately because we would otherwise need to make them entirely illegal,
  // but we want to allow them such that we can run a series of conversion
  // passes to SMT converting different dialects. Also, not marking the
  // unrealized conversion casts legal above (but adding the simplification
  // patterns) does not work, because the dialect conversion framework adds
  // IRRewrite patterns to replace values which are only applied after all
  // operations are legalized. This means, folding the casts away will not be
  // possible in many cases (especially the explicitly inserted target
  // materializations in the lowering of the 'miter' operation).
  RewritePatternSet cleanupPatterns(&getContext());
  populateReconcileUnrealizedCastsPatterns(cleanupPatterns);

  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(cleanupPatterns))))
    return signalPassFailure();
}
