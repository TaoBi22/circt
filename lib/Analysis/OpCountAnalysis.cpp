//===- OpCountAnalysis.cpp - operation count analyses -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the op count analysis.  This is an analysis that
// provides information about the frequency of different kinds of operations
// found in a builtin.module.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OpCountAnalysis.h"
#include "mlir/IR/Operation.h"

using namespace circt;
using namespace analysis;

OpCountAnalysis::OpCountAnalysis(Operation *moduleOp,
                                 mlir::AnalysisManager &am) {
  edgeCount = 0;
  moduleOp->walk([&](Operation *op) {
    auto opName = op->getName();
    // Update opCounts
    opCounts[opName]++;

    // Update operandCounts
    operandCounts[opName][op->getNumOperands()]++;

    for (auto result: op->getResults()) {
      edgeCount += std::distance(result.getUses().begin(), result.getUses().end());
    }
    for (auto &region: op->getRegions()) {
      for (auto &block: region.getBlocks()) {
        for (auto &arg: block.getArguments()){
          edgeCount += std::distance(arg.getUses().begin(), arg.getUses().end());
        }
      }
    }
  });
}

SmallVector<OperationName> OpCountAnalysis::getFoundOpNames() {
  SmallVector<OperationName> opNames;
  for (auto pair : opCounts)
    opNames.push_back(pair.first);
  return opNames;
}

size_t OpCountAnalysis::getOpCount(OperationName opName) {
  return opCounts[opName];
}

size_t OpCountAnalysis::getEdgeCount() {
  return edgeCount;
}

DenseMap<size_t, size_t>
OpCountAnalysis::getOperandCountMap(OperationName opName) {
  return operandCounts[opName];
}
