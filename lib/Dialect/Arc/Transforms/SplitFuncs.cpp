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
  llvm::outs() << "numops " << numOps << "\n";
  llvm::outs() << "numblocks " << numBlocks << "\n";
  OpBuilder opBuilder(funcOp->getContext());
  std::vector<Block *> blocks;
  assert(funcOp->getNumRegions() == 1);
  Block *frontBlock = &(funcOp.getBody().front());
  blocks.push_back(frontBlock);
  for (int i = 0; i < numBlocks - 1; i++) {
    // auto *block = opBuilder.createBlock(&(funcOp.getBody().front()));
    std::vector<Location> locs;
    for (auto t : frontBlock->getArgumentTypes()) {
      locs.push_back(funcOp.getLoc());
    }
    auto *block = opBuilder.createBlock(&(funcOp.getBody()), {},
                                        frontBlock->getArgumentTypes(), locs);
    blocks.push_back(block);
  }
  // Reverse the block list so that order of blocks in vector matches order in
  // region
  // std::reverse(blocks.begin(), blocks.end());

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
    // // ReturnOps go to their own block for liveness analysis
    // if (isa<ReturnOp>(op)) {
    //   returnBlock->push_back(&op);
    // } else {
    (*blockIter)->push_back(&op);
  }
  // Transfer arguments to the final block
  // funcOp->dump();
  // for (auto block : blocks) {
  //   block->dump();
  // }
  // return success();

  funcOp.dump();

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

  // for (auto oldArg : frontBlock->getArguments()) {
  //   // auto newArg = blocks.back()->addArgument(oldArg.getType(),
  //   // funcOp.getLoc());
  //   replaceAllUsesInRegionWith(oldArg, newArg, funcOp.getBody());
  // }
  // frontBlock->eraseArguments(0, frontBlock->getNumArguments());

  // Create funcs to contain blocks
  // Block *currentBlock = blocks[0];
  // Block *previousBlock;
  std::vector<Operation *> funcs;
  // auto liveOut = liveness.getLiveOut(blocks[0]);
  auto liveOut = liveness.getLiveIn(blocks[0]);
  blocks[0]->dump();
  llvm::outs() << "ELS:\n";
  for (auto el : liveOut) {
    llvm::outs() << "EL:\n";

    el.dump();
  }
  Liveness::ValueSetT liveIn;
  for (int i = 0; i < blocks.size() - 1; i++) {
    // previousBlock = currentBlock;
    liveIn = liveOut;
    Block *currentBlock = blocks[i];
    liveOut = liveness.getLiveOut(currentBlock);
    std::vector<Type> inTypes(blocks.back()->getArgumentTypes().begin(),
                              blocks.back()->getArgumentTypes().end());
    std::vector<Value> inValues(blocks.back()->getArguments().begin(),
                                blocks.back()->getArguments().end());
    // llvm::for_each(liveIn, [&inTypes, &inValues, &argMap](auto el) {
    //   auto argLookup = argMap.find(el);
    //   if (argLookup != argMap.end()) {
    //     inValues.push_back(argLookup->second);
    //     inTypes.push_back(argLookup->second.getType());
    //   } else {
    //     inValues.push_back(el);
    //     inTypes.push_back(el.getType());
    //   }
    // });
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
    opBuilder.setInsertionPointToEnd(funcOp->getBlock());
    auto funcName = funcOp.getName().str() + std::to_string(i);
    auto funcType =
        opBuilder.getFunctionType(TypeRange(inTypes), TypeRange(outTypes));
    auto newFunc = opBuilder.create<FuncOp>(
        funcOp->getLoc(), funcName,
        opBuilder.getFunctionType({inTypes}, {outTypes}));
    auto funcBlock = newFunc.addEntryBlock();
    for (auto &op : make_early_inc_range(currentBlock->getOperations())) {
      op.remove();
      funcBlock->push_back(&op);
    }
    funcs.push_back(newFunc);
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
    opBuilder.setInsertionPointToStart(blocks[i + 1]);
    if (funcName == "Simple1") {
      for (auto el : inValues) {
        el.dump();
      }
    }
    Operation *callOp = opBuilder.create<func::CallOp>(
        funcOp->getLoc(), outTypes, funcName, ValueRange(inValues));
    auto callResults = callOp->getResults();
    for (int j = 0; j < outValues.size(); j++) {
      // TODO: this will affect all as of yet unmoved blocks, which is bad
      // (maybe)!
      // llvm::outs() << outValues.size();
      // llvm::outs() << callResults.size();
      assert(callResults.size() == outValues.size());
      // for (auto user : outValues[i].getUsers()) {
      //   if (user->getBlock() == previousBlock) {
      //   }
      // }
      assert(outValues[j] != callResults[j]);
      // funcOp.dump();
      for (auto &use : outValues[j].getUses()) {
      }
      replaceAllUsesInRegionWith(outValues[j], callResults[j],
                                 funcOp.getBody());
      //   replaceAllUsesInRegionWith(outValues[j], callResults[j],
      //                              *previousBlock->getParent());
      //   funcOp.dump();
      //   for (auto user : outValues[j].getUsers()) {
      //     if (user->getBlock() != currentBlock) {
      //       llvm::outs() << "Use remaining";
      //       user->getBlock()->dump();
      //     }
      //     assert(user->getBlock() == currentBlock);
      //   }
      //   previousBlock->getParent()->getParentOp()->dump();
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
  // }dump

  // for (auto block : blocks)
  //   block->getParentOp()->dump();
  // llvm::outs() << "andfunc";
  for (auto func : funcs) {
    func->dump();
  }
  funcOp->dump();
  // llvm::outs() << liveness.getLiveIn(returnBlock).size();
  // llvm::for_each(make_early_inc_range(returnBlock->getOperations()),
  //                [&blocks](Operation &op) {
  //                  op.remove();
  //                  blocks[0]->push_back(&op);
  //                  for (auto def : op.getOperands()) {
  //                    def.dump();
  //                    def.getDefiningOp()->dump();
  //                    def.getDefiningOp()->getParentOp()->dump();
  //                  }
  //                });
  // returnBlock->erase();

  return success();
}

std::unique_ptr<Pass> arc::createSplitFuncsPass() {
  return std::make_unique<SplitFuncsPass>();
}
