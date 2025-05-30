//===-- RTGPasses.td - RTG pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the passes that operate on the RTG dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD
#define CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD

include "mlir/Pass/PassBase.td"

def ElaborationPass : Pass<"rtg-elaborate", "mlir::ModuleOp"> {
  let summary = "elaborate the randomization parts";
  let description = [{
    This pass interprets most RTG operations to perform the represented
    randomization and in the process get rid of those operations. This means,
    after this pass the IR does not contain any random constructs within tests
    anymore.
  }];

  let options = [
    Option<"seed", "seed", "unsigned", /*default=*/"",
      "The seed for any RNG constructs used in the pass.">,
  ];

  let dependentDialects = ["mlir::index::IndexDialect"];
}

def EmitRTGISAAssemblyPass : Pass<"rtg-emit-isa-assembly", "mlir::ModuleOp"> {
  let summary = "Elaborate the contexts of RTG";
  let description = [{
    Emits all 'rtg.test's in the IR in a format understood by assemblers.

    There are two options to specify lists of instructions that are not
    supported by the assembler. For instructions in any of those lists, this
    pass will emit the equivalent binary representation.

    This pass operates on the `InstructionOpInterface` and folds constant-like
    operations to support downstream dialects.
  }];

  let options = [
    Option<"splitOutput", "split-output", "bool", /*default=*/"false",
      "If 'true' emits one file per 'rtg.test' in the IR. The name of the "
      "file matches the test name and is placed in 'path'. Otherwise, path "
      "is interpreted as the full file path including filename.">,
    Option<"path", "path", "std::string", /*default=*/"",
      "The directory or file path in which the output files should be "
      "created. If empty is is emitted to stderr (not allowed if "
      "'split-output' is set to 'true')">,
    Option<"unsupportedInstructionsFile", "unsupported-instructions-file",
      "std::string", /*default=*/"",
      "An absolute path to a file with a list of instruction names not "
      "supported by the assembler.">,
    ListOption<"unsupportedInstructions", "unsupported-instructions",
      "std::string",
      "A list of ISA instruction names not supported by the assembler.">,
  ];
}

def InlineSequencesPass : Pass<"rtg-inline-sequences", "mlir::ModuleOp"> {
  let summary = "inline and interleave sequences";
  let description = [{
    Inline all sequences into tests and remove the 'rtg.sequence' operations.
    Also computes and materializes all interleaved sequences
    ('interleave_sequences' operation).
  }];

  let statistics = [
    Statistic<"numSequencesInlined", "num-sequences-inlined",
      "Number of sequences inlined into another sequence or test.">,
    Statistic<"numSequencesInterleaved", "num-sequences-interleaved",
      "Number of sequences interleaved with another sequence.">,
  ];
}

def LinearScanRegisterAllocationPass : Pass<
    "rtg-linear-scan-register-allocation", "rtg::TestOp"> {

  let summary = "simple linear scan register allocation for RTG";
  let description = [{
    Performs a simple version of the linear scan register allocation algorithm
    based on the 'rtg.virtual_reg' operations.

    This pass is expected to be run after elaboration.
  }];

  let statistics = [
    Statistic<"numRegistersSpilled", "num-registers-spilled",
      "Number of registers spilled to the stack.">,
  ];
}

def LowerUniqueLabelsPass : Pass<"rtg-lower-unique-labels", "mlir::ModuleOp"> {
  let summary = "lower label_unique_decl to label_decl operations";
  let description = [{
    This pass lowers label_unique_decl operations to label_decl operations by
    creating a unique label string based on all the existing unique and
    non-unique label declarations in the module.
  }];

  let statistics = [
    Statistic<"numLabelsLowered", "num-labels-lowered",
      "Number of unique labels lowered to regular label declarations.">,
  ];
}

#endif // CIRCT_DIALECT_RTG_TRANSFORMS_RTGPASSES_TD
