//===- circt-bmc.cpp - The circt-bmc bounded model checker ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'circt-bmc' tool
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToSMT.h"
#include "circt/Conversion/HWToBTOR2.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Conversion/SMTToZ3LLVM.h"
#include "circt/Conversion/VerifToSMT.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Emit/EmitDialect.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "circt/Support/Passes.h"
#include "circt/Support/Version.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#ifdef CIRCT_BMC_ENABLE_JIT
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "llvm/Support/TargetSelect.h"
#endif

namespace cl = llvm::cl;

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Command-line options declaration
//===----------------------------------------------------------------------===//

static cl::OptionCategory mainCategory("circt-bmc Options");

static cl::opt<std::string> inputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input file>"),
                                          cl::cat(mainCategory));

static llvm::cl::opt<std::string>
    directory("export-dir",
              llvm::cl::desc("Directory path to write the files to."),
              llvm::cl::init("./"));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true), cl::cat(mainCategory));

//===----------------------------------------------------------------------===//
// Tool implementation
//===----------------------------------------------------------------------===//

/// This function initializes the various components of the tool and
/// orchestrates the work to be done.
static LogicalResult executeBMC(MLIRContext &context) {
  // Create the timing manager we use to sample execution times.
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  auto ts = tm.getRootScope();

  OwningOpRef<ModuleOp> module;
  {
    auto parserTimer = ts.nest("Parse MLIR input");
    // Parse the provided input files.
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  }
  if (!module)
    return failure();

  PassManager pm(&context);
  pm.enableVerifier(verifyPasses);
  pm.enableTiming(ts);
  if (failed(applyPassManagerCLOptions(pm)))
    return failure();

  // Separate formal contracts into other files

  pm.addPass(verif::createLowerContractsPass());

  if (failed(pm.run(module.get())))
    return failure();

  // Copy each contract to new MLIR context and lower
  for (auto [j, op] : llvm::enumerate(module->getOps<verif::FormalOp>())) {
    auto newContext =
        std::make_unique<MLIRContext>(context.getDialectRegistry());
    auto newModule = ModuleOp::create(UnknownLoc::get(newContext.get()));
    newModule.push_back(op->clone());
    op->remove();
    PassManager pm(newContext.get());
    std::string fileName = "contract" + std::to_string(j) + ".btor2";
    SmallString<128> filePath(directory);
    llvm::sys::path::append(filePath, fileName);
    std::string errorMessage;
    auto output = mlir::openOutputFile(filePath, &errorMessage);
    pm.addPass(verif::createLowerFormalToHWPass());
    pm.addPass(circt::createConvertHWToBTOR2Pass(output->os()));
    if (failed(pm.run(newModule)))
      return failure();
    output->keep();
  }

  // TODO - this errors out on symbolic_value ops in the module. I think the
  // solution to this is to add a lowering to HWToBTOR2 Dump top module in file
  module.get()->dump();
  pm.clear();
  std::string fileName = "top.btor2";
  SmallString<128> filePath(directory);
  llvm::sys::path::append(filePath, fileName);
  std::string errorMessage;
  auto output = mlir::openOutputFile(filePath, &errorMessage);
  pm.addPass(circt::createSimpleCanonicalizerPass());
  pm.addPass(verif::createLowerFormalToHWPass());
  pm.addPass(circt::createConvertHWToBTOR2Pass(output->os()));
  if (failed(pm.run(module.get())))
    return failure();
  output->keep();

  // std::string fileName = symbolOp.getName().str() + ".h";
  // SmallString<128> filePath(directory);
  // llvm::sys::path::append(filePath, fileName);
  // std::string errorMessage;
  // auto output = mlir::openOutputFile(filePath, &errorMessage);
  // if (!output)
  //   return module.emitError(errorMessage);
  return success();
}

/// The entry point for the `circt-bmc` tool:
/// configures and parses the command-line options,
/// registers all dialects within a MLIR context,
/// and calls the `executeBMC` function to do the actual work.
int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Hide default LLVM options, other than for this tool.
  // MLIR options are added below.
  cl::HideUnrelatedOptions(mainCategory);

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerAsmPrinterCLOptions();
  cl::AddExtraVersionPrinter(
      [](llvm::raw_ostream &os) { os << circt::getCirctVersion() << '\n'; });

  // Parse the command-line options provided by the user.
  cl::ParseCommandLineOptions(
      argc, argv,
      "circt-bmc - bounded model checker\n\n"
      "\tThis tool checks all possible executions of a hardware module up to a "
      "given time bound to check whether any asserted properties can be "
      "violated.\n");

  // Set the bug report message to indicate users should file issues on
  // llvm/circt and not llvm/llvm-project.
  llvm::setBugReportMsg(circt::circtBugReportMsg);

  // Register the supported CIRCT dialects and create a context to work with.
  DialectRegistry registry;
  // clang-format off
  registry.insert<
    circt::comb::CombDialect,
    circt::emit::EmitDialect,
    circt::hw::HWDialect,
    circt::om::OMDialect,
    circt::seq::SeqDialect,
    circt::smt::SMTDialect,
    circt::verif::VerifDialect,
    mlir::arith::ArithDialect,
    mlir::BuiltinDialect,
    mlir::func::FuncDialect,
    mlir::LLVM::LLVMDialect,
    mlir::scf::SCFDialect
  >();
  // clang-format on
  mlir::func::registerInlinerExtension(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  MLIRContext context(registry);

  // Setup of diagnostic handling.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  // Avoid printing a superfluous note on diagnostic emission.
  context.printOpOnDiagnostic(false);

  // Perform the logical equivalence checking; using `exit` to avoid the slow
  // teardown of the MLIR context.
  exit(failed(executeBMC(context)));
}
