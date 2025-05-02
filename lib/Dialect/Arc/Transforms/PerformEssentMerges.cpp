//===- PerformEssentMerges.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "arc-perform-essent-merges"

namespace circt {
namespace arc {
#define GEN_PASS_DEF_PERFORMESSENTMERGES
#include "circt/Dialect/Arc/ArcPasses.h.inc"
} // namespace arc
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace arc;

class PerformEssentMergesAnalysis {
public:
  PerformEssentMergesAnalysis(ModuleOp module, OpBuilder b, int threshold)
      : threshold(threshold), b(b) {
    module->walk([&](Operation *op) {
      if (auto defOp = dyn_cast<DefineOp>(op)) {
        arcDefs[defOp.getNameAttr()] = &defOp;
      } else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        auto arcName = cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee())
                           .getLeafReference();
        if (arcName)
          arcCalls[arcName].push_back(callOp);
      }
    });
  }

  bool canMergeArcs(CallOpInterface firstArc, CallOpInterface secondArc);
  llvm::LogicalResult mergeArcs(CallOpInterface firstArc,
                                CallOpInterface secondArc);

private:
  DenseMap<StringAttr, DefineOp *> arcDefs;
  DenseMap<StringAttr, SmallVector<CallOpInterface>> arcCalls;
  int threshold;
  OpBuilder b;
};

bool PerformEssentMergesAnalysis::canMergeArcs(CallOpInterface firstArc,
                                               CallOpInterface secondArc) {
  // Make sure ops either have same latency or entirely consume each other.
  if (isa<StateOp>(firstArc) || isa<StateOp>(secondArc)) {
    for (auto res : firstArc->getResults()) {
      for (auto *user : res.getUsers()) {
        if (user != secondArc.getOperation()) {
          return false;
        }
      }
    }
  }
  // Make sure arcs do not have other, conflicting uses.
  auto firstArcName = cast<mlir::SymbolRefAttr>(firstArc.getCallableForCallee())
                          .getLeafReference();
  auto secondArcName =
      cast<mlir::SymbolRefAttr>(secondArc.getCallableForCallee())
          .getLeafReference();
  llvm::outs() << arcCalls[firstArcName].size();
  return arcCalls[firstArcName].size() <= 1 ||
         arcCalls[secondArcName].size() <= 1;
}

llvm::LogicalResult
PerformEssentMergesAnalysis::mergeArcs(CallOpInterface firstArc,
                                       CallOpInterface secondArc) {
  // Check we're actually able to merge the arcs
  if (!canMergeArcs(firstArc, secondArc))
    return llvm::failure();
  auto firstArcName = cast<mlir::SymbolRefAttr>(firstArc.getCallableForCallee())
                          .getLeafReference();
  auto secondArcName =
      cast<mlir::SymbolRefAttr>(secondArc.getCallableForCallee())
          .getLeafReference();
  auto *firstArcDefine = arcDefs[firstArcName];
  auto *secondArcDefine = arcDefs[secondArcName];

  auto firstArcResults = firstArc->getResults();
  auto secondArcArgs = secondArc->getOperands();

  // Generate mapping from second arc's values to first arc's
  IRMapping mapping;
  auto firstArcOutputs =
      firstArcDefine->getBodyBlock().getTerminator()->getOperands();
  // Check which arguments to the second arc need corresponding arguments added,
  // and which are just inputs from the first arc anyway, so we can just map
  // them to the corresponding values the first arc outputs
  for (auto [argi, arg] : llvm::enumerate(secondArcArgs)) {
    bool needToAddArg = true;
    for (auto [resi, res] : llvm::enumerate(firstArcResults)) {
      if (arg == res) {
        // Values are passed between arcs so can just map directly
        mapping.map(secondArcDefine->getArgument(argi), firstArcOutputs[resi]);
        needToAddArg = false;
      }
    }
    if (needToAddArg) {
      auto newArg = firstArcDefine->getBodyBlock().addArgument(arg.getType(),
                                                               arg.getLoc());
      mapping.map(secondArcDefine->getArgument(argi), newArg);
    }
  }
  // TODO: get rid of firstArc outputs which were only consumed by the second
  // arc

  // Prepare operands for new terminator and delete existing terminators
  auto secondArcOutputs =
      secondArcDefine->getBodyBlock().getTerminator()->getOperands();
  SmallVector<Value> newOutputs;
  newOutputs.append(firstArcOutputs.begin(), firstArcOutputs.end());
  newOutputs.append(secondArcOutputs.begin(), secondArcOutputs.end());
  firstArcDefine->getBodyBlock().getTerminator()->erase();
  secondArcDefine->getBodyBlock().getTerminator()->erase();
  secondArcDefine->getBody().cloneInto(&firstArcDefine->getBody(), mapping);
  // Now create a new terminator
  b.setInsertionPointToEnd(&firstArcDefine->getBodyBlock());
  b.create<OutputOp>(firstArcDefine->getLoc(), ValueRange(newOutputs));
  // Update inputs of call to first arc
  // Update output signature of call to first arc
  // Replace the second call's outputs with the first call's
  for (auto [index, res] : llvm::enumerate(secondArc->getResults())) {
    res.replaceAllUsesWith(firstArc->getResult(index + firstArcOutputs.size()));
  }
  // Combine latencies
  // Delete old second arc
  secondArc->erase();
  secondArcDefine->erase();
  return success();
}

struct PerformEssentMergesPass
    : public arc::impl::PerformEssentMergesBase<PerformEssentMergesPass> {
  using PerformEssentMergesBase::PerformEssentMergesBase;

  void runOnOperation() override;
};

void PerformEssentMergesPass::runOnOperation() {
  OpBuilder b(getOperation());
  auto analysis =
      PerformEssentMergesAnalysis(getOperation(), b, optimalPartitionSize);
  auto ops = getOperation().getOps<hw::HWModuleOp>();
  auto modOp = *ops.begin();
  auto callOps = modOp.getOps<CallOp>();
  auto x = callOps.begin();
  x++;
  auto firstOp = *callOps.begin();
  auto secondOp = *x;
  analysis.mergeArcs(firstOp, secondOp);
}

std::unique_ptr<Pass> arc::createPerformEssentMergesPass() {
  return std::make_unique<PerformEssentMergesPass>();
}
