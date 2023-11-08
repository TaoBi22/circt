//===- SplitFuncs.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "arc-split-funcs"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_SPLITFUNCS
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;
using namespace hw;
using namespace func;
using mlir::OpTrait::ConstantLike;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct SplitFuncsPass : public arc::impl::SplitFuncsBase<SplitFuncsPass> {
  SplitFuncsPass() = default;
  SplitFuncsPass(const SplitFuncsPass &pass) : SplitFuncsPass() {}

  void runOnOperation() override;
  LogicalResult lowerModel(ModelOp modelOp);
  LogicalResult lowerFunc(FuncOp funcOp);

  SymbolTable *symbolTable;
  int splitBound;

  Statistic numFuncsCreated{this, "funcs-created",
                            "Number of new functions created"};

  // using SplitFuncsBase::funcSizeThreshold;
};
} // namespace

// void SplitFuncsPass::runOnOperation() {
//   // symbolTable = &getAnalysis<SymbolTable>();
//   lowerFunc(getOperation());
//   // for (auto op : getOperation().getOps<ModelOp>()) {
//   //   if (failed(lowerModel(op)))
//   //     return signalPassFailure();
//   // }
// }

void SplitFuncsPass::runOnOperation() {
  symbolTable = &getAnalysis<SymbolTable>();
  for (auto op : getOperation().getOps<ModelOp>()) {
    if (failed(lowerModel(op)))
      return signalPassFailure();
    op->dump();
  }
}

LogicalResult SplitFuncsPass::lowerModel(ModelOp modelOp) {
  SmallVector<FuncOp> funcOps(modelOp.getOps<FuncOp>());
  for (auto op : funcOps) {
    if (failed(lowerFunc(op)))
      return failure();
  }
  return success();
}

LogicalResult SplitFuncsPass::lowerFunc(FuncOp funcOp) {
  int funcSizeThreshold = 2;
  int numOps =
      funcOp->getRegion(0).front().getOperations().size(); // TODO neaten this!
  int numBlocks = ceil(numOps / funcSizeThreshold);
  OpBuilder opBuilder(funcOp->getContext());
  std::vector<Block *> blocks;
  assert(funcOp->getNumRegions() == 1);
  Block *frontBlock = &(funcOp.getBody().front());
  blocks.push_back(frontBlock);
  for (int i = 0; i < numBlocks - 1; i++) {
    std::vector<Location> locs;
    for (auto t : frontBlock->getArgumentTypes()) {
      locs.push_back(funcOp.getLoc());
    }
    auto *block = opBuilder.createBlock(&(funcOp.getBody()), {},
                                        frontBlock->getArgumentTypes(), locs);
    blocks.push_back(block);
  }

  int numOpsInBlock = 0;
  std::vector<Block *>::iterator blockIter = blocks.begin();
  for (auto &op : llvm::make_early_inc_range(*frontBlock)) {
    if (numOpsInBlock >= funcSizeThreshold) {
      blockIter++;
      numOpsInBlock = 0;
      opBuilder.setInsertionPointToEnd(*blockIter);
    }
    numOpsInBlock++;
    // Don't bother moving ops to the original block
    if (*blockIter == (frontBlock))
      continue;
    // Remove op from original block and insert in new block
    op.remove();
    (*blockIter)->push_back(&op);
  }

  DenseMap<Value, Value> argMap;
  // Move function arguments to the block that will stay in the function
  for (int argIndex = 0; argIndex < frontBlock->getNumArguments(); argIndex++) {
    auto oldArg = frontBlock->getArgument(argIndex);
    auto newArg = blocks.back()->getArgument(argIndex);
    replaceAllUsesInRegionWith(oldArg, newArg, funcOp.getBody());
    argMap.insert(std::pair(oldArg, newArg));
  }
  Liveness liveness(funcOp);

  funcOp.dump();
  std::vector<Operation *> funcs;
  auto liveOut = liveness.getLiveIn(blocks[0]);
  Liveness::ValueSetT liveIn;
  auto argTypes = blocks.back()->getArgumentTypes();
  auto args = blocks.back()->getArguments();
  for (int i = 0; i < blocks.size() - 1; i++) {
    liveIn = liveOut;
    Block *currentBlock = blocks[i];
    liveOut = liveness.getLiveOut(currentBlock);
    std::vector<Type> outTypes;
    std::vector<Value> outValues;
    llvm::for_each(liveOut, [&outTypes, &outValues, &argMap](auto el) {
      auto argLookup = argMap.find(el);
      if (argLookup != argMap.end()) {
        outValues.push_back(argLookup->second);
        outTypes.push_back(argLookup->second.getType());
      } else {
        outValues.push_back(el);
        outTypes.push_back(el.getType());
      }
    });
    opBuilder.setInsertionPoint(funcOp);
    SmallString<32> funcName;
    funcName.append(funcOp.getName());
    funcName.append("_");
    funcName.append(std::to_string(i));
    // auto funcName = funcOp.getName().str() + std::to_string(i);
    auto newFunc =
        opBuilder.create<FuncOp>(funcOp->getLoc(), funcName,
                                 opBuilder.getFunctionType(argTypes, outTypes));
    symbolTable->insert(newFunc);
    auto *funcBlock = newFunc.addEntryBlock();
    for (auto &op : make_early_inc_range(currentBlock->getOperations())) {
      op.remove();
      funcBlock->push_back(&op);
    }
    funcs.push_back(newFunc);
    currentBlock->erase();
    currentBlock = funcBlock;
    int j = 0;
    for (auto el : args) {
      replaceAllUsesInRegionWith(el, newFunc.getArgument(j++),
                                 newFunc.getRegion());
    }
    opBuilder.setInsertionPointToEnd(currentBlock);
    opBuilder.create<ReturnOp>(funcOp->getLoc(), ValueRange(outValues));
    opBuilder.setInsertionPointToStart(blocks[i + 1]);
    Operation *callOp = opBuilder.create<func::CallOp>(
        funcOp->getLoc(), outTypes, funcName, args);
    auto callResults = callOp->getResults();
    for (int k = 0; k < outValues.size(); k++) {
      replaceAllUsesInRegionWith(outValues[k], callResults[k],
                                 funcOp.getBody());
    }
  }
  funcOp.dump();
  return success();
}

std::unique_ptr<Pass> arc::createSplitFuncsPass() {
  return std::make_unique<SplitFuncsPass>();
}
