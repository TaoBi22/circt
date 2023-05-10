//===- LowerIntrinsics.cpp - Lower Intrinsics -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerIntrinsics pass.  This pass processes FIRRTL
// extmodules with intrinsic annotations and rewrites the instances as
// appropriate.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace firrtl;

// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerIntrinsicsPass : public LowerIntrinsicsBase<LowerIntrinsicsPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static ParseResult hasNPorts(StringRef name, FModuleLike mod, unsigned n) {
  if (mod.getPorts().size() != n) {
    mod.emitError(name) << " has " << mod.getPorts().size()
                        << " ports instead of " << n;
    return failure();
  }
  return success();
}

static ParseResult namedPort(StringRef name, FModuleLike mod, unsigned n,
                             StringRef portName) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].getName().equals(portName)) {
    mod.emitError(name) << " port " << n << " named '" << ports[n].getName()
                        << "' instead of '" << portName << "'";
    return failure();
  }
  return success();
}

template <typename T>
static ParseResult typedPort(StringRef name, FModuleLike mod, unsigned n) {
  auto ports = mod.getPorts();
  if (n >= ports.size()) {
    mod.emitError(name) << " missing port " << n;
    return failure();
  }
  if (!ports[n].type.isa<T>()) {
    mod.emitError(name) << " port " << n << " not of correct type";
    return failure();
  }
  return success();
}

template <typename T>
static ParseResult sizedPort(StringRef name, FModuleLike mod, unsigned n,
                             int32_t size) {
  auto ports = mod.getPorts();
  if (failed(typedPort<T>(name, mod, n)))
    return failure();
  if (ports[n].type.cast<T>().getWidth() != size) {
    mod.emitError(name) << " port " << n << " not size " << size;
    return failure();
  }
  return success();
}

static ParseResult hasNParam(StringRef name, FModuleLike mod, unsigned n) {
  unsigned num = 0;
  if (mod.getParameters())
    num = mod.getParameters().size();
  if (n != num) {
    mod.emitError(name) << " has " << num << " parameters instead of " << n;
    return failure();
  }
  return success();
}

static ParseResult namedParam(StringRef name, FModuleLike mod,
                              StringRef paramName) {
  for (auto a : mod.getParameters()) {
    auto param = a.cast<ParamDeclAttr>();
    if (param.getName().getValue().equals(paramName)) {
      if (param.getValue().isa<StringAttr>())
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be a string but is not";
      return failure();
    }
  }
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static ParseResult namedIntParam(StringRef name, FModuleLike mod,
                                 StringRef paramName) {
  for (auto a : mod.getParameters()) {
    auto param = a.cast<ParamDeclAttr>();
    if (param.getName().getValue().equals(paramName)) {
      if (param.getValue().isa<IntegerAttr>())
        return success();

      mod.emitError(name) << " has parameter '" << param.getName()
                          << "' which should be an integer but is not";
      return failure();
    }
  }
  mod.emitError(name) << " is missing parameter " << paramName;
  return failure();
}

static InstanceGraphNode *lookupInstNode(InstancePathCache &instancePathCache,
                                         FModuleLike mod) {
  // Seems like you should be able to use a dyn_cast here, but alas
  if (isa<FIntModuleOp>(mod))
    return instancePathCache.instanceGraph[cast<FIntModuleOp>(mod)];
  return instancePathCache.instanceGraph[cast<FExtModuleOp>(mod)];
}

static bool lowerCirctSizeof(InstancePathCache &instancePathCache,
                             FModuleLike mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.sizeof", mod, 2) ||
      namedPort("circt.sizeof", mod, 0, "i") ||
      namedPort("circt.sizeof", mod, 1, "size") ||
      sizedPort<UIntType>("circt.sizeof", mod, 1, 32) ||
      hasNParam("circt.sizeof", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputWire = builder.create<WireOp>(ports[0].type).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<SizeOfIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
  }
  return true;
}

static bool lowerCirctIsX(InstancePathCache &instancePathCache,
                          FModuleLike mod) {
  auto ports = mod.getPorts();
  if (hasNPorts("circt.isX", mod, 2) || namedPort("circt.isX", mod, 0, "i") ||
      namedPort("circt.isX", mod, 1, "found") ||
      sizedPort<UIntType>("circt.isX", mod, 1, 1) ||
      hasNParam("circt.isX", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto inputWire = builder.create<WireOp>(ports[0].type).getResult();
    inst.getResult(0).replaceAllUsesWith(inputWire);
    auto size = builder.create<IsXIntrinsicOp>(inputWire);
    inst.getResult(1).replaceAllUsesWith(size);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPlusArgTest(InstancePathCache &instancePathCache,
                                  FModuleLike mod) {
  if (hasNPorts("circt.plusargs.test", mod, 1) ||
      namedPort("circt.plusargs.test", mod, 0, "found") ||
      sizedPort<UIntType>("circt.plusargs.test", mod, 0, 1) ||
      hasNParam("circt.plusargs.test", mod, 1) ||
      namedParam("circt.plusargs.test", mod, "FORMAT"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsTestIntrinsicOp>(
        param.getValue().cast<StringAttr>());
    inst.getResult(0).replaceAllUsesWith(newop);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPlusArgValue(InstancePathCache &instancePathCache,
                                   FModuleLike mod) {
  if (hasNPorts("circt.plusargs.value", mod, 2) ||
      namedPort("circt.plusargs.value", mod, 0, "found") ||
      namedPort("circt.plusargs.value", mod, 1, "result") ||
      sizedPort<UIntType>("circt.plusargs.value", mod, 0, 1) ||
      hasNParam("circt.plusargs.value", mod, 1) ||
      namedParam("circt.plusargs.value", mod, "FORMAT"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto newop = builder.create<PlusArgsValueIntrinsicOp>(
        inst.getResultTypes(), param.getValue().cast<StringAttr>());
    inst.getResult(0).replaceAllUsesWith(newop.getFound());
    inst.getResult(1).replaceAllUsesWith(newop.getResult());
    inst.erase();
  }
  return true;
}

static bool lowerCirctClockGate(InstancePathCache &instancePathCache,
                                FModuleLike mod) {
  if (hasNPorts("circt.clock_gate", mod, 3) ||
      namedPort("circt.clock_gate", mod, 0, "in") ||
      namedPort("circt.clock_gate", mod, 1, "en") ||
      namedPort("circt.clock_gate", mod, 2, "out") ||
      typedPort<ClockType>("circt.clock_gate", mod, 0) ||
      sizedPort<UIntType>("circt.clock_gate", mod, 1, 1) ||
      typedPort<ClockType>("circt.clock_gate", mod, 2) ||
      hasNParam("circt.clock_gate", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto en = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    inst.getResult(1).replaceAllUsesWith(en);
    auto out = builder.create<ClockGateIntrinsicOp>(in, en, Value{});
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

template <class Op>
static bool lowerCirctPropAssertLike(InstancePathCache &instancePathCache,
                                     FModuleLike mod) {
  if (hasNPorts("circt.prop.assert", mod, 3) ||
      namedPort("circt.prop.assert", mod, 0, "prop") ||
      namedPort("circt.prop.assert", mod, 1, "clock") ||
      namedPort("circt.prop.assert", mod, 2, "disable_iff") ||
      sizedPort<UIntType>("circt.prop.assert", mod, 0, 1) ||
      typedPort<ClockType>("circt.prop.assert", mod, 1) ||
      sizedPort<UIntType>("circt.prop.assert", mod, 2, 1) ||
      hasNParam("circt.prop.assert", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto prop = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto clock =
        builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    auto disable_iff =
        builder.create<WireOp>(inst.getResult(2).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(prop);
    inst.getResult(1).replaceAllUsesWith(clock);
    inst.getResult(2).replaceAllUsesWith(disable_iff);
    builder.create<Op>(prop, clock, disable_iff);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPropAnd(InstancePathCache &instancePathCache,
                              FModuleLike mod) {
  if (hasNPorts("circt.prop.and", mod, 3) ||
      namedPort("circt.prop.and", mod, 0, "lhs") ||
      namedPort("circt.prop.and", mod, 1, "rhs") ||
      namedPort("circt.prop.and", mod, 2, "out") ||
      sizedPort<UIntType>("circt.prop.and", mod, 0, 1) ||
      sizedPort<UIntType>("circt.prop.and", mod, 1, 1) ||
      sizedPort<UIntType>("circt.prop.and", mod, 2, 1) ||
      hasNParam("circt.prop.and", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<PropAndIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPropImpl(InstancePathCache &instancePathCache,
                               FModuleLike mod) {
  if (hasNPorts("circt.prop.impl", mod, 3) ||
      namedPort("circt.prop.impl", mod, 0, "lhs") ||
      namedPort("circt.prop.impl", mod, 1, "rhs") ||
      namedPort("circt.prop.impl", mod, 2, "out") ||
      sizedPort<UIntType>("circt.prop.impl", mod, 0, 1) ||
      sizedPort<UIntType>("circt.prop.impl", mod, 1, 1) ||
      sizedPort<UIntType>("circt.prop.impl", mod, 2, 1) ||
      hasNParam("circt.prop.impl", mod, 1) ||
      namedIntParam("circt.prop.impl", mod, "overlap"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  auto overlap = param.getValue().cast<IntegerAttr>().getValue() == 1;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out =
        builder.create<PropImplIntrinsicOp>(lhs.getType(), lhs, rhs, overlap);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctPropEventually(InstancePathCache &instancePathCache,
                                     FModuleLike mod) {
  if (hasNPorts("circt.prop.eventually", mod, 2) ||
      namedPort("circt.prop.eventually", mod, 0, "in") ||
      namedPort("circt.prop.eventually", mod, 1, "out") ||
      sizedPort<UIntType>("circt.prop.eventually", mod, 0, 1) ||
      sizedPort<UIntType>("circt.prop.eventually", mod, 1, 1) ||
      hasNParam("circt.prop.eventually", mod, 1) ||
      namedIntParam("circt.prop.eventually", mod, "strong"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  auto strong = param.getValue().cast<IntegerAttr>().getValue() == 1;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out =
        builder.create<PropEventuallyIntrinsicOp>(in.getType(), in, strong);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctSeqAnd(InstancePathCache &instancePathCache,
                             FModuleLike mod) {
  if (hasNPorts("circt.seq.and", mod, 3) ||
      namedPort("circt.seq.and", mod, 0, "lhs") ||
      namedPort("circt.seq.and", mod, 1, "rhs") ||
      namedPort("circt.seq.and", mod, 2, "out") ||
      sizedPort<UIntType>("circt.seq.and", mod, 0, 1) ||
      sizedPort<UIntType>("circt.seq.and", mod, 1, 1) ||
      sizedPort<UIntType>("circt.seq.and", mod, 2, 1) ||
      hasNParam("circt.seq.and", mod, 0))
    return false;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<SeqAndIntrinsicOp>(lhs.getType(), lhs, rhs);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctSeqDelayUnary(InstancePathCache &instancePathCache,
                                    FModuleLike mod) {
  if (hasNPorts("circt.seq.delay.unary", mod, 2) ||
      namedPort("circt.seq.delay.unary", mod, 0, "in") ||
      namedPort("circt.seq.delay.unary", mod, 1, "out") ||
      sizedPort<UIntType>("circt.seq.delay.unary", mod, 0, 1) ||
      sizedPort<UIntType>("circt.seq.delay.unary", mod, 1, 1) ||
      hasNParam("circt.seq.delay.unary", mod, 1) ||
      namedIntParam("circt.seq.delay.unary", mod, "N"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  auto cycles = param.getValue().cast<IntegerAttr>().getValue() == 1;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto in = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(in);
    auto out =
        builder.create<SeqDelayUnaryIntrinsicOp>(in.getType(), in, cycles);
    inst.getResult(1).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

static bool lowerCirctSeqDelayBinary(InstancePathCache &instancePathCache,
                                     FModuleLike mod) {
  if (hasNPorts("circt.seq.delay.binary", mod, 3) ||
      namedPort("circt.seq.delay.binary", mod, 0, "lhs") ||
      namedPort("circt.seq.delay.binary", mod, 1, "rhs") ||
      namedPort("circt.seq.delay.binary", mod, 2, "out") ||
      sizedPort<UIntType>("circt.seq.delay.binary", mod, 0, 1) ||
      sizedPort<UIntType>("circt.seq.delay.binary", mod, 1, 1) ||
      sizedPort<UIntType>("circt.seq.delay.binary", mod, 2, 1) ||
      hasNParam("circt.seq.delay.binary", mod, 1) ||
      namedIntParam("circt.seq.delay.binary", mod, "N"))
    return false;

  auto param = mod.getParameters()[0].cast<ParamDeclAttr>();
  auto cycles = param.getValue().cast<IntegerAttr>().getValue() == 1;

  for (auto *use : lookupInstNode(instancePathCache, mod)->uses()) {
    auto inst = cast<InstanceOp>(use->getInstance().getOperation());
    ImplicitLocOpBuilder builder(inst.getLoc(), inst);
    auto lhs = builder.create<WireOp>(inst.getResult(0).getType()).getResult();
    auto rhs = builder.create<WireOp>(inst.getResult(1).getType()).getResult();
    inst.getResult(0).replaceAllUsesWith(lhs);
    inst.getResult(1).replaceAllUsesWith(rhs);
    auto out = builder.create<SeqDelayBinaryIntrinsicOp>(lhs.getType(), lhs,
                                                         rhs, cycles);
    inst.getResult(2).replaceAllUsesWith(out);
    inst.erase();
  }
  return true;
}

std::pair<const char *, std::function<bool(InstancePathCache &, FModuleLike)>>
    intrinsics[] = {
        {"circt.sizeof", lowerCirctSizeof},
        {"circt_sizeof", lowerCirctSizeof},
        {"circt.isX", lowerCirctIsX},
        {"circt_isX", lowerCirctIsX},
        {"circt.plusargs.test", lowerCirctPlusArgTest},
        {"circt_plusargs_test", lowerCirctPlusArgTest},
        {"circt.plusargs.value", lowerCirctPlusArgValue},
        {"circt_plusargs_value", lowerCirctPlusArgValue},
        {"circt.clock_gate", lowerCirctClockGate},
        {"circt_clock_gate", lowerCirctClockGate},
        {"circt.prop.assert", lowerCirctPropAssertLike<AssertIntrinsicOp>},
        {"circt_prop_assert", lowerCirctPropAssertLike<AssertIntrinsicOp>},
        {"circt.prop.assume", lowerCirctPropAssertLike<AssumeIntrinsicOp>},
        {"circt_prop_assume", lowerCirctPropAssertLike<AssumeIntrinsicOp>},
        {"circt.prop.cover", lowerCirctPropAssertLike<CoverIntrinsicOp>},
        {"circt_prop_cover", lowerCirctPropAssertLike<CoverIntrinsicOp>},
        {"circt.prop.and", lowerCirctPropAnd},
        {"circt_prop_and", lowerCirctPropAnd},
        {"circt.prop.impl", lowerCirctPropImpl},
        {"circt_prop_impl", lowerCirctPropImpl},
        {"circt.prop.eventually", lowerCirctPropEventually},
        {"circt_prop_eventually", lowerCirctPropEventually},
        {"circt.seq.and", lowerCirctSeqAnd},
        {"circt_seq_and", lowerCirctSeqAnd},
        {"circt.seq.delay.unary", lowerCirctSeqDelayUnary},
        {"circt_seq_delay_unary", lowerCirctSeqDelayUnary},
        {"circt.seq.delay.binary", lowerCirctSeqDelayBinary},
        {"circt_seq_delay_binary", lowerCirctSeqDelayBinary},
};

// This is the main entrypoint for the lowering pass.
void LowerIntrinsicsPass::runOnOperation() {
  size_t numFailures = 0;
  size_t numConverted = 0;
  InstancePathCache instancePathCache(getAnalysis<InstanceGraph>());
  for (auto &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (!isa<FExtModuleOp, FIntModuleOp>(op))
      continue;
    StringAttr intname;
    if (isa<FExtModuleOp>(op)) {
      auto anno = AnnotationSet(&op).getAnnotation("circt.Intrinsic");
      if (!anno)
        continue;
      intname = anno.getMember<StringAttr>("intrinsic");
      if (!intname) {
        op.emitError("intrinsic annotation with no intrinsic name");
        ++numFailures;
        continue;
      }
    } else {
      intname = cast<FIntModuleOp>(op).getIntrinsicAttr();
      if (!intname) {
        op.emitError("intrinsic module with no intrinsic name");
        ++numFailures;
        continue;
      }
    }

    bool found = false;
    for (const auto &intrinsic : intrinsics) {
      if (intname.getValue().equals(intrinsic.first)) {
        found = true;
        if (intrinsic.second(instancePathCache, cast<FModuleLike>(op))) {
          ++numConverted;
          op.erase();
        } else {
          ++numFailures;
        }
        break;
      }
    }
    if (!found) {
      op.emitError("unknown intrinsic: '") << intname.getValue() << "'";
      ++numFailures;
    }
  }
  if (numFailures)
    signalPassFailure();
  if (!numConverted)
    markAllAnalysesPreserved();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerIntrinsicsPass() {
  return std::make_unique<LowerIntrinsicsPass>();
}
