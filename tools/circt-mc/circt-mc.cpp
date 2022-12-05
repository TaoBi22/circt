//===- circt-mc.cpp - The circt-mc model checker --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-mc' tool
//
//===----------------------------------------------------------------------===//

#include "LogicExporter.h"
#include "Solver.h"
#include "Circuit.h"
#include "Utility.h"
#include "circt/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"\
#include "llvm/Support/CommandLine.h"
#include <z3++.h>

namespace cl = llvm::cl;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-lec Options");

static cl::opt<std::string>
    moduleName1("c1",
                cl::desc("Specify a named module to verify properties over."),
                cl::value_desc("module name"), cl::cat(mainCategory));

static cl::opt<std::string> inputFileName(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static mlir::LogicalResult checkProperty(mlir::MLIRContext &context,
                                         int bound) {

  mlir::OwningOpRef<mlir::ModuleOp> inputFile =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFileName, &context);
  if (!inputFile)
    return mlir::failure();

  // Create solver and add circuit
  Solver s(&context);
  Solver::Circuit *circuitModel = s.addCircuit(inputFileName, true);

  mlir::PassManager pm(&context);
  pm.addPass(std::make_unique<LogicExporter>(moduleName1, circuitModel));
  mlir::ModuleOp m = inputFile.get();
  if (failed(pm.run(m)))
    return mlir::failure();

  // TODO: load property constraints
  //circuitModel->loadProperty();

  // Set initial state of model
  circuitModel->setInitialState();

  for (int i = 0; i < bound; i++) {
    if (circuitModel->checkCycle()) {
      circuitModel->updateInputs(i);
    } else {
      return mlir::failure();
    }
  }
  lec::outs << "Success!\n";
  return mlir::success();
}

int main(int argc, char **argv) {
  // Configure the relevant command-line options.
  cl::HideUnrelatedOptions(mainCategory);
  mlir::registerMLIRContextCLOptions();

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-lec - logical equivalence checker\n\n"
      "\tThis tool compares two input circuit descriptions to determine whether"
      " they are logically equivalent.\n");

  // Register all the CIRCT dialects and create a context to work with.
  mlir::DialectRegistry registry;
  circt::registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  exit(failed(checkProperty(context, 1000)));
}
