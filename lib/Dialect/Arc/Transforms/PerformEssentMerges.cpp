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
#include "mlir/Bytecode/Encoding.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

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

class ArcEssentMerger {
public:
  ArcEssentMerger(ModuleOp module, IRRewriter &r, int threshold)
      : threshold(threshold), r(r), module(module) {
    regenerateArcMapping();
  }

  bool canMergeArcs(CallOpInterface firstArc, CallOpInterface secondArc);
  llvm::LogicalResult mergeArcs(CallOpInterface firstArc,
                                CallOpInterface secondArc);
  llvm::LogicalResult applySingleParentMerges();
  llvm::LogicalResult applySmallSiblingMerges();
  llvm::LogicalResult applySmallIntoBigSiblingMerges();

private:
  DenseMap<StringAttr, DefineOp> arcDefs;
  DenseMap<StringAttr, SmallVector<CallOpInterface>> arcCalls;
  int threshold;
  IRRewriter &r;
  ModuleOp module;
  bool isSmall(CallOpInterface arc);
  void regenerateArcMapping();
};

void ArcEssentMerger::regenerateArcMapping() {
  module->walk([&](Operation *op) {
    if (auto defOp = dyn_cast<DefineOp>(op)) {
      arcDefs[defOp.getNameAttr()] = defOp;
    } else if (auto callOp = dyn_cast<CallOpInterface>(op)) {
      auto arcName = cast<mlir::SymbolRefAttr>(callOp.getCallableForCallee())
                         .getLeafReference();
      if (arcName)
        arcCalls[arcName].push_back(callOp);
    }
  });
}

bool ArcEssentMerger::canMergeArcs(CallOpInterface firstArc,
                                   CallOpInterface secondArc) {
  // Make sure first arc doesn't use second arc (probably a FIXME, could make
  // direction dynamic)
  auto secondArcUsers = secondArc->getUsers();
  if (std::find(secondArcUsers.begin(), secondArcUsers.end(), firstArc) !=
      secondArcUsers.end()) {
    return false;
  }
  // Make sure ops either have same latency or entirely
  // consume each other.
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
  return arcCalls[firstArcName].size() <= 1 &&
         arcCalls[secondArcName].size() <= 1;
}

bool ArcEssentMerger::isSmall(CallOpInterface arc) {
  // Check if the arc is small enough to be merged
  auto arcName =
      cast<mlir::SymbolRefAttr>(arc.getCallableForCallee()).getLeafReference();
  auto arcDef = arcDefs[arcName];
  int numOps = 0;
  arcDef->walk([&](Operation *op) { numOps++; });
  // Decrement by one before comparison to account for the operation itself
  return --numOps < threshold;
}

llvm::LogicalResult ArcEssentMerger::mergeArcs(CallOpInterface firstArc,
                                               CallOpInterface secondArc) {
  // Check we're actually able to merge the arcs

  if (!canMergeArcs(firstArc, secondArc))
    return llvm::failure();

  auto firstArcName = cast<mlir::SymbolRefAttr>(firstArc.getCallableForCallee())
                          .getLeafReference();
  auto secondArcName =
      cast<mlir::SymbolRefAttr>(secondArc.getCallableForCallee())
          .getLeafReference();
  LLVM_DEBUG(llvm::dbgs() << "Merging arcs " << firstArcName << " and "
                          << secondArcName << "\n");
  auto firstArcDefine = arcDefs[firstArcName];
  auto secondArcDefine = arcDefs[secondArcName];

  SmallVector<Value> firstArcArgs;
  if (auto firstState = dyn_cast<StateOp>(firstArc.getOperation())) {
    firstArcArgs.append(firstState.getInputs().begin(),
                        firstState.getInputs().end());
  } else {
    firstArcArgs.append(firstArc->getOperands().begin(),
                        firstArc->getOperands().end());
  }
  auto firstArcResults = firstArc->getResults();
  SmallVector<Value> secondArcArgs;
  if (auto secondState = dyn_cast<StateOp>(secondArc.getOperation())) {
    secondArcArgs.append(secondState.getInputs().begin(),
                         secondState.getInputs().end());
  } else {
    secondArcArgs.append(secondArc->getOperands().begin(),
                         secondArc->getOperands().end());
  }
  auto secondArcResults = secondArc->getResults();

  // Generate mapping from second arc's values to first arc's
  IRMapping mapping;
  auto firstArcOutputs =
      firstArcDefine.getBodyBlock().getTerminator()->getOperands();
  // Check which arguments to the second arc need corresponding arguments added,
  // and which are just inputs from the first arc anyway, so we can just map
  // them to the corresponding values the first arc outputs
  SmallVector<Value> argReplacements;
  // The arguments to the second arc which are staying as arguments (so need to
  // be fed to the new call)
  SmallVector<Value> additionalCallOperands;
  for (auto [argi, arg] : llvm::enumerate(secondArcArgs)) {
    bool needToAddArg = true;
    for (auto [resi, res] : llvm::enumerate(firstArcResults)) {
      if (arg == res) {
        // Values are passed between arcs so can just map directly
        argReplacements.push_back(firstArcOutputs[resi]);
        // mapping.map(secondArcDefine.getArgument(argi),
        // firstArcOutputs[resi]);
        needToAddArg = false;
      }
    }
    if (needToAddArg) {
      firstArcDefine.insertArgument(firstArcDefine.getNumArguments(),
                                    arg.getType(), {}, arg.getLoc());

      argReplacements.push_back(firstArcDefine.getArguments().back());
      additionalCallOperands.push_back(arg);
      // mapping.map(secondArcDefine.getArgument(argi), newArg);
    }
  }
  // FIXME: get rid of firstArc outputs which were only consumed by the
  // second arc

  // Prepare operands for new terminator and delete existing terminators
  SmallVector<Value> newOutputs;

  newOutputs.append(firstArcOutputs.begin(), firstArcOutputs.end());

  // TODO: this needs a value range, not a mapping - need to figure that one out
  r.inlineBlockBefore(&secondArcDefine.getBodyBlock(),
                      firstArcDefine.getBodyBlock().getTerminator(),
                      argReplacements);
  // inlineBlockBefore deletes original values so we need to fetch the new
  // values from the second block's terminator (which is now the penultimate op
  // in the block)

  auto *newSecondArcTerminator =
      firstArcDefine.getBodyBlock().getTerminator()->getPrevNode();

  auto secondArcOutputs = newSecondArcTerminator->getOperands();
  SmallVector<Type> secondArcOutputTypes;

  for (auto res : secondArcOutputs) {
    secondArcOutputTypes.push_back(res.getType());
  }
  newOutputs.append(secondArcOutputs.begin(), secondArcOutputs.end());
  // // before we delete the new terminator we also need to check whether the
  // first
  // // arc consumes any values from the second
  // for (auto [operandIndex, operand] :
  // llvm::enumerate(firstArc->getOperands()))
  //   for (auto [resultIndex, res] : llvm::enumerate(secondArc->getResults()))
  //     if (res == operand)
  //       r.replaceAllUsesWith(firstArcDefine.getArgument(operandIndex),
  //                            newSecondArcTerminator->getOperand(resultIndex));

  newSecondArcTerminator->erase();
  // Now create a new terminator
  firstArcDefine.getBodyBlock().getTerminator()->erase();
  r.setInsertionPointToEnd(&firstArcDefine.getBodyBlock());
  r.create<OutputOp>(firstArcDefine->getLoc(), ValueRange(newOutputs));
  // Update firstArc results

  auto originalResultCount = firstArc->getNumResults();
  for (auto [index, resTy] : llvm::enumerate(secondArcOutputTypes)) {
    firstArcDefine.insertResult(index + originalResultCount, resTy, {});
  }

  // Update inputs of call to first arc
  // Update output signature of call to first arc
  // Replace the second call's outputs with the first call's
  // Make a smallvec with all the inputs to our new call op
  SmallVector<Value> newCallOperands;
  newCallOperands.append(firstArcArgs.begin(), firstArcArgs.end());
  newCallOperands.append(additionalCallOperands.begin(),
                         additionalCallOperands.end());
  r.setInsertionPointAfter(firstArc);

  int totalLatency = 0;
  Value clock;
  bool mustBeState = false;
  // TODO: these clock checks are only safe if we check clock equivalence in our
  // canMergeArcs function
  if (auto firstState = dyn_cast<StateOp>(firstArc.getOperation())) {
    mustBeState = true;
    totalLatency += firstState.getLatency();
    clock = firstState.getClock();
  }
  if (auto secondState = dyn_cast<StateOp>(secondArc.getOperation())) {
    mustBeState = true;
    totalLatency += secondState.getLatency();
    clock = secondState.getClock();
  }

  Operation *newCall;
  if (mustBeState) {
    newCall = r.create<StateOp>(firstArc->getLoc(), firstArcDefine, clock,
                                Value(), totalLatency,
                                ValueRange(newCallOperands), ValueRange());
  } else {
    newCall = r.create<CallOp>(firstArc->getLoc(), firstArcDefine,
                               ValueRange(newCallOperands));
  }

  for (auto [index, res] : llvm::enumerate(firstArcResults))
    res.replaceAllUsesWith(newCall->getResult(index));
  for (auto [index, res] : llvm::enumerate(secondArcResults)) {
    res.replaceAllUsesWith(
        newCall->getResult(index + firstArc->getNumResults()));
  }

  // Combine latencies
  // Delete redundant call op
  firstArc.erase();
  // Delete old second arc
  secondArc->erase();
  secondArcDefine->erase();
  return success();
}

llvm::LogicalResult ArcEssentMerger::applySingleParentMerges() {
  // First, gather the list of merges:
  bool changed = true;
  int iterationsRemaining = 10000;
  while (changed) {
    if (--iterationsRemaining < 0) {
      return failure();
    }
    // TODO: this should probably be a greedy pattern rewriter of some sort for
    // upstreaming
    changed = false;
    SmallVector<std::pair<CallOpInterface, CallOpInterface>> arcsToMerge;
    module->walk<WalkOrder::PreOrder>([&](CallOpInterface callOp) {
      if (!isa<StateOp, CallOp>(callOp.getOperation()))
        return;
      // Check if op has single parent
      llvm::DenseSet<Operation *> parents;

      for (auto operand : callOp->getOperands()) {

        if (isa<BlockArgument>(operand))
          return;

        parents.insert(operand.getDefiningOp());
      }

      // We're only interested if this op has one parent and it's an arc call
      if (parents.empty() || parents.size() > 1 ||
          !isa<StateOp, CallOp>(*parents.begin())) {

        return;
      }

      // For now we'll just always merge these, but I need to check in the
      // ESSENT code whether this should be restricted to just small partitions
      arcsToMerge.push_back(
          std::pair(cast<CallOpInterface>(*parents.begin()), callOp));
      // mergeArcs(cast<CallOpInterface>(*parents.begin()), callOp);
    });

    // Now perform all the merges we can
    // We can't merge arcs we've already merged this cycle - save them for next
    // time
    SmallVector<CallOpInterface> touchedArcs;
    for (auto [firstArc, secondArc] : arcsToMerge) {
      // ignore any merges involving arcs we've already merged
      if (std::find(touchedArcs.begin(), touchedArcs.end(), firstArc) !=
          touchedArcs.end()) {
        continue;
      }
      if (std::find(touchedArcs.begin(), touchedArcs.end(), secondArc) !=
          touchedArcs.end()) {
        continue;
      }
      changed |= llvm::succeeded(mergeArcs(firstArc, secondArc));
      touchedArcs.push_back(firstArc);
      touchedArcs.push_back(secondArc);
    }
    regenerateArcMapping();
  }
  return llvm::success();
}

llvm::LogicalResult ArcEssentMerger::applySmallSiblingMerges() {
  // iterate to fixed point
  bool changed = true;
  while (changed) {
    changed = false;
    // First, gather sets of small siblings
    // Just do this with a nested for loop - could do this with a bitmap but
    // that would involve some massive APInts so very memory hungry
    // Vector of pairs - first element is a list of parents, second is a list of
    // siblings. We need to store the parents so we can invalidate merges that
    // we've tampered with the parents of (actually do we??? they'll still be
    // siblings after a merge???)
    SmallVector<
        std::pair<DenseSet<CallOpInterface>, SmallVector<CallOpInterface>>>
        smallSiblingSets;
    SmallVector<CallOpInterface> alreadyGrouped;
    module->walk([&](CallOpInterface callOp) {
      // We only care if the arc is small
      if (!isSmall(callOp))
        return;
      // Check it's not already in a grouping
      if (std::find(alreadyGrouped.begin(), alreadyGrouped.end(), callOp) !=
          alreadyGrouped.end())
        return;
      // Collect the set of parents
      DenseSet<CallOpInterface> parents;
      for (auto operand : callOp->getOperands()) {
        // We only care about ops with only arc parents
        if (!isa<BlockArgument>(operand) &&
            isa<CallOp, StateOp>(operand.getDefiningOp())) {

          parents.insert(cast<CallOpInterface>(operand.getDefiningOp()));
        }
      }
      // TODO: should we ignore parentless arcs here? Are they siblings?
      if (parents.empty())
        return;
      // Look for small siblings
      SmallVector<CallOpInterface> siblings;
      module->walk([&](CallOpInterface secondCallOp) {
        // We only care if the arc is small
        if (!isSmall(secondCallOp))
          return;
        // No need to check if it's grouped as the first arc would be too
        // Check if the two arcs are siblings
        DenseSet<CallOpInterface> secondParents;
        for (auto operand : secondCallOp->getOperands()) {
          // We only care about ops with only arc parents
          if (!isa<BlockArgument>(operand) &&
              isa<CallOp, StateOp>(operand.getDefiningOp()))
            secondParents.insert(
                cast<CallOpInterface>(operand.getDefiningOp()));
        }
        if (parents != secondParents) {
          return;
        }
        siblings.push_back(secondCallOp);
      });
      if (siblings.empty())
        return;
      // Otherwise we have a set of siblings! Yay!
      siblings.push_back(callOp);
      smallSiblingSets.push_back(std::pair(parents, siblings));
      alreadyGrouped.append(siblings.begin(), siblings.end());
    });
    // Now we have a set of small siblings, we can merge them
    // TODO: work out how ESSENT does this exactly
    // For now we'll just merge them all into the first one
    for (auto [parents, siblings] : smallSiblingSets) {
      // Can't merge if we don't have at least two siblings
      if (siblings.size() < 2)
        continue;
      // WIP better technique
      // We want to reduce cut edges - each child in common is a cut edge, so we
      // can maximize common children
      SmallVector<CallOpInterface> pairedSiblings;
      for (auto sibling : siblings) {
        // Check if this sibling is already paired
        if (std::find(pairedSiblings.begin(), pairedSiblings.end(), sibling) !=
            pairedSiblings.end()) {
          continue;
        }
        int bestNumCutEdges = -1;
        int bestReductionIndex = -1;
        for (auto [candIndex, candidateSibling] : llvm::enumerate(siblings)) {
          // Can't pair a sibling with itself, don't be so silly!
          if (sibling == candidateSibling) {
            continue;
          }
          // Check if this sibling is already paired
          if (std::find(pairedSiblings.begin(), pairedSiblings.end(),
                        candidateSibling) != pairedSiblings.end()) {
            continue;
          }
          // Calculate reduction in number of cut edges
          int numCutEdges = 0;
          for (auto user : sibling->getUsers()) {
            if (user == candidateSibling) {
              numCutEdges++;
              continue;
            }
            for (auto candidateUser : candidateSibling->getUsers()) {
              if (candidateUser == sibling) {
                numCutEdges++;
                continue;
              }
              if (user == candidateUser) {
                numCutEdges++;
                continue;
              }
            }
          }
          if (numCutEdges > bestNumCutEdges) {
            bestNumCutEdges = numCutEdges;
            bestReductionIndex = candIndex;
          }
        }
        // If there are no possible merging candidates then we know there are no
        // more merges to do
        if (bestReductionIndex == -1) {
          break;
        }
        changed |=
            llvm::succeeded(mergeArcs(sibling, siblings[bestReductionIndex]));
      }
    }
  }
  regenerateArcMapping();
  return llvm::success();
}

llvm::LogicalResult ArcEssentMerger::applySmallIntoBigSiblingMerges() {
  // iterate to fixed point
  bool changed = true;
  while (changed) {
    changed = false;
    // First pair is small siblings, second is big siblings
    SmallVector<
        std::pair<SmallVector<CallOpInterface>, SmallVector<CallOpInterface>>>
        splitSiblingSets;

    SmallVector<CallOpInterface> alreadyGrouped;
    module->walk([&](CallOpInterface callOp) {
      // We only care if the starting arc is small (so we avoid fully big
      // partitioning levels)
      if (!isSmall(callOp))
        return;
      // Check it's not already in a grouping
      if (std::find(alreadyGrouped.begin(), alreadyGrouped.end(), callOp) !=
          alreadyGrouped.end())
        return;
      // Collect the set of parents
      DenseSet<CallOpInterface> parents;
      for (auto operand : callOp->getOperands()) {
        // We only care about ops with only arc parents
        if (!isa<BlockArgument>(operand) &&
            isa<CallOp, StateOp>(operand.getDefiningOp()))
          parents.insert(cast<CallOpInterface>(operand.getDefiningOp()));
      }
      // TODO: should we ignore parentless arcs here? Are they siblings?
      if (parents.empty())
        return;
      // Look for small siblings
      SmallVector<CallOpInterface> smallSiblings;
      SmallVector<CallOpInterface> bigSiblings;
      module->walk([&](CallOpInterface secondCallOp) {
        // No need to check if it's grouped as the first arc would be too
        // Check if the two arcs are siblings
        DenseSet<CallOpInterface> secondParents;
        for (auto operand : secondCallOp->getOperands()) {
          // We only care about ops with only arc parents
          if (!isa<BlockArgument>(operand) &&
              isa<CallOp, StateOp>(operand.getDefiningOp()))
            secondParents.insert(
                cast<CallOpInterface>(operand.getDefiningOp()));
        }
        if (parents != secondParents)
          return;
        if (isSmall(secondCallOp))
          smallSiblings.push_back(secondCallOp);
        else
          bigSiblings.push_back(secondCallOp);
      });
      if (smallSiblings.empty() && bigSiblings.empty())
        return;
      // Otherwise we have a set of siblings! Yay!
      smallSiblings.push_back(callOp);
      splitSiblingSets.push_back(std::pair(smallSiblings, bigSiblings));
      alreadyGrouped.append(smallSiblings.begin(), smallSiblings.end());
      alreadyGrouped.append(bigSiblings.begin(), bigSiblings.end());
    });
    // Now we have a set of small siblings, we can merge them
    // TODO: work out how ESSENT does this exactly
    // For now we'll just merge them all into the first one
    for (auto [smallSiblings, bigSiblings] : splitSiblingSets) {
      SmallVector<CallOpInterface> mergedBigSiblings;
      for (auto smallSibling : smallSiblings) {
        // Merge all the siblings into the first one
        for (auto bigSibling : bigSiblings) {
          if (std::find(mergedBigSiblings.begin(), mergedBigSiblings.end(),
                        bigSibling) != mergedBigSiblings.end()) {
            // TODO: this might cause segfaults if the bigsibling is being
            // messed with during iteration - maybe need to precompute this
            changed |= llvm::succeeded(mergeArcs(bigSibling, smallSibling));
            mergedBigSiblings.push_back(bigSibling);
          }
        }
        mergedBigSiblings.push_back(smallSibling);
      }
    }
  }
  regenerateArcMapping();
  return llvm::success();
}

struct PerformEssentMergesPass
    : public arc::impl::PerformEssentMergesBase<PerformEssentMergesPass> {
  using PerformEssentMergesBase::PerformEssentMergesBase;

  void runOnOperation() override;
};

void PerformEssentMergesPass::runOnOperation() {
  IRRewriter r(getOperation()->getContext());
  auto merger = ArcEssentMerger(getOperation(), r, optimalPartitionSize);
  if (failed(merger.applySingleParentMerges()))
    return signalPassFailure();
  if (failed(merger.applySmallSiblingMerges()))
    return signalPassFailure();
  if (failed(merger.applySmallIntoBigSiblingMerges()))
    return signalPassFailure();
}

std::unique_ptr<Pass> arc::createPerformEssentMergesPass() {
  return std::make_unique<PerformEssentMergesPass>();
}
