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

void SplitFuncsPass::runOnOperation() {
  // symbolTable = &getAnalysis<SymbolTable>();
  lowerFunc(getOperation());
  // for (auto op : getOperation().getOps<ModelOp>()) {
  //   if (failed(lowerModel(op)))
  //     return signalPassFailure();
  // }
}

LogicalResult SplitFuncsPass::lowerModel(ModelOp modelOp) {
  for (auto op : modelOp.getOps<FuncOp>()) {
    if (failed(lowerFunc(op)))
      return failure();
  }
}

LogicalResult SplitFuncsPass::lowerFunc(FuncOp funcOp) {
  int funcSizeThreshold = 2;
  int numOps =
      funcOp->getRegion(0).front().getOperations().size(); // TODO neaten this!
  int numBlocks = ceil(numOps / funcSizeThreshold);
  OpBuilder opBuilder(funcOp->getContext());
  std::vector<Block *> blocks;
  assert(funcOp->getNumRegions() == 1);
  blocks.push_back(&(funcOp->getRegion(0).front()));
  for (int i = 0; i < numBlocks - 1; i++) {
    auto *block = opBuilder.createBlock(&funcOp->getRegion(0));
    blocks.push_back(block);
  }
  auto returnBlock = opBuilder.createBlock(&funcOp->getRegion(0));

  int numOpsInBlock = 0;
  std::vector<Block *>::iterator blockIter = blocks.begin();
  for (auto &op : llvm::make_early_inc_range(funcOp.getBody().front())) {
    if (numOpsInBlock >= funcSizeThreshold) {
      blockIter++;
      numOpsInBlock = 0;
      opBuilder.setInsertionPointToEnd(*blockIter);
    }
    numOpsInBlock++;
    // Don't bother moving ops to the original block
    if (*blockIter == &(funcOp->getRegion(0).front()))
      continue;
    // Remove op from original block and insert in new block
    op.remove();
    // ReturnOps go to their own block for liveness analysis
    if (isa<ReturnOp>(op)) {
      returnBlock->push_back(&op);
    } else {
      (*blockIter)->push_back(&op);
    }
  }

  Liveness liveness(funcOp);
  // Create funcs to contain blocks
  Block *currentBlock = blocks[0];
  Block *previousBlock;
  auto liveOut = liveness.getLiveOut(currentBlock);
  Liveness::ValueSetT liveIn;
  for (int i = 1; i < blocks.size(); i++) {
    previousBlock = currentBlock;
    liveIn = liveOut;
    currentBlock = blocks[i];
    liveOut = liveness.getLiveOut(currentBlock);
    std::vector<Type> inTypes;
    std::vector<Value> inValues;
    llvm::for_each(liveIn, [&inTypes, &inValues](auto el) {
      inTypes.push_back(el.getType());
      inValues.push_back(el);
    });
    std::vector<Type> outTypes;
    std::vector<Value> outValues;
    llvm::for_each(liveOut, [&outTypes, &outValues](auto el) {
      outTypes.push_back(el.getType());
      outValues.push_back(el);
    });
    opBuilder.setInsertionPointAfter(funcOp);
    auto funcName = funcOp.getName().str() + std::to_string(i);
    auto newFunc = opBuilder.create<FuncOp>(
        funcOp->getLoc(), funcName,
        opBuilder.getFunctionType({inTypes}, {outTypes}));
    auto funcBlock = newFunc.addEntryBlock();
    for (auto &op : make_early_inc_range(currentBlock->getOperations())) {
      op.remove();
      funcBlock->push_back(&op);
    }
    currentBlock->erase();
    currentBlock = funcBlock;
    int j = 0;
    for (auto el : inValues) {
      replaceAllUsesInRegionWith(el, newFunc.getArgument(j++),
                                 newFunc.getRegion());
    }
    opBuilder.setInsertionPointToEnd(currentBlock);
    Operation *returnOp =
        opBuilder.create<ReturnOp>(funcOp->getLoc(), ValueRange(outValues));
    // auto prevReturns = (*previousBlock).getOps<ReturnOp>();
    // if (prevReturns.empty()) {
    //   opBuilder.setInsertionPointToEnd(previousBlock);
    // } else {
    //   opBuilder.setInsertionPoint(*prevReturns.begin());
    // }
    opBuilder.setInsertionPoint(&previousBlock->back());
    Operation *callOp = opBuilder.create<func::CallOp>(
        funcOp->getLoc(), outTypes, funcName, ValueRange(inValues));
    auto callResults = callOp->getResults();
    for (int j = 0; j < outValues.size(); j++) {
      // TODO: this will affect all as of yet unmoved blocks, which is bad
      // (maybe)!
      llvm::outs() << outValues.size();
      llvm::outs() << callResults.size();
      assert(callResults.size() == outValues.size());
      // for (auto user : outValues[i].getUsers()) {
      //   if (user->getBlock() == previousBlock) {
      //   }
      // }
      assert(outValues[j] != callResults[j]);
      funcOp.dump();
      replaceAllUsesInRegionWith(outValues[j], callResults[j],
                                 funcOp.getBody());
      replaceAllUsesInRegionWith(outValues[j], callResults[j],
                                 *previousBlock->getParent());
      funcOp.dump();
      // for (auto user : outValues[j].getUsers()) {
      //   if (user->getBlock() != currentBlock) {
      //     llvm::outs() << "Use remaining";
      //     user->getBlock()->dump();
      //   }
      //   assert(user->getBlock() == currentBlock);
      // }
      // previousBlock->getParent()->getParentOp()->dump();
    }
  }
  /**/
  // llvm::outs() << "cf\n";
  // currentBlock = blocks[0];
  // *previousBlock;
  // liveOut = liveness.getLiveOut(currentBlock);
  // liveIn = liveOut;
  // for (int i = 1; i < blocks.size(); i++) {
  //   llvm::outs() << "nextBlock\n";
  //   previousBlock = currentBlock;
  //   liveIn = liveOut;
  //   currentBlock = blocks[i];
  //   liveOut = liveness.getLiveOut(currentBlock);
  //   // llvm::outs() << liveOut << "ok\n";
  //   for (auto el : liveOut) {
  //     el.dump();
  //     llvm::outs() << "and done\n";
  //   }
  //   std::vector<Value> outValues;
  //   llvm::for_each(liveOut, [&outValues](auto el) { outValues.push_back(el);
  //   }); for (auto el : outValues) {
  //     el.dump();
  //     llvm::outs() << "and ho!\n";
  //   }

  //   auto callOp = previousBlock->getOps<func::CallOp>().begin();
  //   auto results = (*callOp)->getResults();
  //   for (int j = 0; j < outValues.size(); j++) {
  //     llvm::outs() << "REMPLACE\n";
  //     auto tOp = (func::FuncOp)previousBlock->getParentOp();
  //     replaceAllUsesInRegionWith(outValues[i], results[i], tOp.getBody());
  //   }
  // }

  // for (auto block : blocks)
  //   block->getParentOp()->dump();
  // llvm::outs() << "andfunc";
  // funcOp->dump();
  llvm::outs() << liveness.getLiveIn(returnBlock).size();
  llvm::for_each(make_early_inc_range(returnBlock->getOperations()),
                 [&blocks](Operation &op) {
                   op.remove();
                   blocks[0]->push_back(&op);
                   for (auto def : op.getOperands()) {
                     def.dump();
                     def.getDefiningOp()->dump();
                     def.getDefiningOp()->getParentOp()->dump();
                   }
                 });
  returnBlock->erase();

  return success();
}

std::unique_ptr<Pass> arc::createSplitFuncsPass() {
  return std::make_unique<SplitFuncsPass>();
}
