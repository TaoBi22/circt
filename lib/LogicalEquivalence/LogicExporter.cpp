//===- LogicExporter.cpp - class to extrapolate CIRCT IR logic --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the logic-exporting class for the `circt-lec` tool.
///
//===----------------------------------------------------------------------===//

#include "circt/LogicalEquivalence/LogicExporter.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifVisitors.h"
#include "circt/LogicalEquivalence/Circuit.h"
#include "circt/LogicalEquivalence/Solver.h"
#include "circt/LogicalEquivalence/Utility.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "lec-exporter"

using namespace circt;
using namespace mlir;

namespace {

/// This class provides logic-exporting functions for the implemented
/// operations, along with a dispatcher to visit the correct handler.
struct Visitor : public hw::StmtVisitor<Visitor, LogicalResult>,
                 public hw::TypeOpVisitor<Visitor, LogicalResult>,
                 public comb::CombinationalVisitor<Visitor, LogicalResult>,
                 public verif::Visitor<Visitor, LogicalResult> {
  using hw::StmtVisitor<Visitor, LogicalResult>::visitStmt;
  using hw::TypeOpVisitor<Visitor, LogicalResult>::visitTypeOp;
  using comb::CombinationalVisitor<Visitor, LogicalResult>::visitComb;
  friend class hw::StmtVisitor<Visitor, LogicalResult>;
  friend class hw::TypeOpVisitor<Visitor, LogicalResult>;
  friend class comb::CombinationalVisitor<Visitor, LogicalResult>;

  Visitor(Solver::Circuit *circuit) : circuit(circuit) {}
  Solver::Circuit *circuit;

  /// Handles `builtin.module` logic exporting.
  LogicalResult visit(ModuleOp op, llvm::StringRef targetModule) {
    for (auto hwModule : op.getOps<hw::HWModuleOp>()) {
      if (targetModule.empty() || hwModule.getName() == targetModule) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Using module `" << hwModule.getName() << "`\n");
        return visit(hwModule);
      }
    }
    op.emitError("module not found");
    return failure();
  }

  /// Handles `hw.module` logic exporting.
  LogicalResult visit(hw::HWModuleOp op) {
    for (auto argument : op.getBodyBlock()->getArguments())
      circuit->addInput(argument);
    for (auto &op : op.getOps())
      if (failed(dispatch(&op)))
        return failure();
    return success();
  }

  LogicalResult visitUnhandledOp(Operation *op) {
    op->emitOpError("not supported");
    return failure();
  }

  /// Dispatches an operation to the appropriate visit function.
  LogicalResult dispatch(Operation *op) { return dispatchStmtVisitor(op); }

  //===--------------------------------------------------------------------===//
  // hw::StmtVisitor
  //===--------------------------------------------------------------------===//

  LogicalResult visitStmt(hw::InstanceOp op) {
    if (auto hwModule =
            llvm::dyn_cast<hw::HWModuleOp>(op.getReferencedModuleSlow())) {
      circuit->addInstance(op.getInstanceName(), hwModule, op->getOperands(),
                           op->getResults());
      return success();
    }
    op.emitError("instantiated module `" + op.getModuleName() +
                 "` is not an HW module");
    return failure();
  }

  LogicalResult visitStmt(hw::OutputOp op) {
    for (auto operand : op.getOperands())
      circuit->addOutput(operand);
    return success();
  }

  LogicalResult visitInvalidStmt(Operation *op) {
    return dispatchTypeOpVisitor(op);
  }
  LogicalResult visitUnhandledStmt(Operation *op) {
    return visitUnhandledOp(op);
  }

  //===--------------------------------------------------------------------===//
  // hw::TypeOpVisitor
  //===--------------------------------------------------------------------===//

  LogicalResult visitTypeOp(hw::ConstantOp op) {
    circuit->addConstant(op.getResult(), op.getValue());
    return success();
  }

  LogicalResult visitInvalidTypeOp(Operation *op) {
    return dispatchCombinationalVisitor(op);
  }
  LogicalResult visitUnhandledTypeOp(Operation *op) {
    return visitUnhandledOp(op);
  }

  //===--------------------------------------------------------------------===//
  // comb::CombinationalVisitor
  //===--------------------------------------------------------------------===//

  // Visit a comb operation with a variadic number of operands.
  template <typename OpTy, typename FnTy>
  LogicalResult visitVariadicCombOp(OpTy op, FnTy fn) {
    if (!op.getTwoState())
      return op.emitOpError("without 'bin' unsupported");
    (circuit->*fn)(op.getResult(), op.getOperands());
    return success();
  }

  // Visit a comb operation with two operands.
  template <typename OpTy, typename FnTy>
  LogicalResult visitBinaryCombOp(OpTy op, FnTy fn) {
    if (!op.getTwoState())
      return op.emitOpError("without 'bin' unsupported");
    (circuit->*fn)(op.getResult(), op.getLhs(), op.getRhs());
    return success();
  }

  // Visit a comb operation with one operand.
  template <typename OpTy, typename FnTy>
  LogicalResult visitUnaryCombOp(OpTy op, FnTy fn) {
    if (!op.getTwoState())
      return op.emitOpError("without 'bin' unsupported");
    (circuit->*fn)(op.getResult(), op.getInput());
    return success();
  }

  LogicalResult visitComb(comb::AddOp op) {
    return visitVariadicCombOp(op, &Solver::Circuit::performAdd);
  }
  LogicalResult visitComb(comb::AndOp op) {
    return visitVariadicCombOp(op, &Solver::Circuit::performAnd);
  }
  LogicalResult visitComb(comb::ConcatOp op) {
    circuit->performConcat(op.getResult(), op.getOperands());
    return success();
  }
  LogicalResult visitComb(comb::DivSOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performDivS);
  }
  LogicalResult visitComb(comb::DivUOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performDivU);
  }
  LogicalResult visitComb(comb::ExtractOp op) {
    circuit->performExtract(op.getResult(), op.getInput(), op.getLowBit());
    return success();
  }
  LogicalResult visitComb(comb::ICmpOp op) {
    if (!op.getTwoState())
      return op.emitOpError("without 'bin' unsupported");
    circuit->performICmp(op.getResult(), op.getPredicate(), op.getLhs(),
                         op.getRhs());
    return success();
  }
  LogicalResult visitComb(comb::ModSOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performModS);
  }
  LogicalResult visitComb(comb::ModUOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performModU);
  }
  LogicalResult visitComb(comb::MulOp op) {
    return visitVariadicCombOp(op, &Solver::Circuit::performMul);
  }
  LogicalResult visitComb(comb::MuxOp op) {
    if (!op.getTwoState())
      return op.emitOpError("without 'bin' unsupported");
    circuit->performMux(op.getResult(), op.getCond(), op.getTrueValue(),
                        op.getFalseValue());
    return success();
  }
  LogicalResult visitComb(comb::OrOp op) {
    return visitVariadicCombOp(op, &Solver::Circuit::performOr);
  }
  LogicalResult visitComb(comb::ParityOp op) {
    return visitUnaryCombOp(op, &Solver::Circuit::performParity);
  }
  LogicalResult visitComb(comb::ReplicateOp op) {
    circuit->performReplicate(op.getResult(), op.getInput());
    return success();
  }
  LogicalResult visitComb(comb::ShlOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performShl);
  }
  LogicalResult visitComb(comb::ShrSOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performShrS);
  }
  LogicalResult visitComb(comb::ShrUOp op) {
    return visitBinaryCombOp(op, &Solver::Circuit::performShrU);
  }
  LogicalResult visitComb(comb::SubOp op) {
    return visitVariadicCombOp(op, &Solver::Circuit::performSub);
  }
  LogicalResult visitComb(comb::XorOp op) {
    return visitVariadicCombOp(op, &Solver::Circuit::performXor);
  }

  LogicalResult visitInvalidComb(Operation *op) {
    return dispatchVerifVisitor(op);
  }

  LogicalResult visitUnhandledComb(Operation *op) {
    return visitUnhandledOp(op);
  }

  //===--------------------------------------------------------------------===//
  // verif::VerifVisitor
  //===--------------------------------------------------------------------===//

  LogicalResult visitVerif(verif::AssertOp op) {
    if (op.getProperty().getType().isSignlessInteger(1)) {
      circuit->performAssert(op.getProperty());
      return success();
    }
    return op->emitError("LTL properties not yet handled");
  }

  LogicalResult visitVerif(verif::AssumeOp op) {
    if (op.getProperty().getType().isSignlessInteger(1)) {
      circuit->performAssume(op.getProperty());
      return success();
    }
    return op->emitError("LTL properties not yet handled");
  }

  LogicalResult visitVerif(verif::CoverOp op) {
    return op->emitError(
        "Coverage checks not currently supported for model checking.");
  }

  LogicalResult visitInvalidVerif(Operation *op) { return visitSeq(op); }

  LogicalResult visitUnhandledVerif(Operation *op) {
    return visitUnhandledOp(op);
  }

  //===----------------------------------------------------------------------===//
  // Sequential Visitor implementation
  //===----------------------------------------------------------------------===//

  /// Visits seq dialect operations
  mlir::LogicalResult visitSeq(mlir::Operation *op) {
    mlir::LogicalResult outcome =
        llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(op)
            .Case<circt::seq::CompRegOp>([&](circt::seq::CompRegOp &op) {
              return visitSeqOp(op, circuit);
            })
            .Case<circt::seq::FirRegOp>([&](circt::seq::FirRegOp &op) {
              return visitSeqOp(op, circuit);
            })
            .Case<circt::seq::FromClockOp>([&](circt::seq::FromClockOp &op) {
              return visitSeqOp(op, circuit);
            })
            .Default([&](mlir::Operation *op) { return visitUnhandledOp(op); });
    return outcome;
  }

  mlir::LogicalResult visitSeqOp(circt::seq::CompRegOp &op,
                                 Solver::Circuit *circuit) {
    mlir::Value input = op.getInput();
    mlir::Value clk = op.getClk();
    mlir::Value data = op.getData();
    mlir::Value reset = op.getReset();
    mlir::Value resetValue = op.getResetValue();
    circuit->performCompReg(input, clk, data, reset, resetValue);
    return mlir::success();
  }

  mlir::LogicalResult visitSeqOp(circt::seq::FirRegOp &op,
                                 Solver::Circuit *circuit) {
    assert(!op.getIsAsync() && "Async resets not supported.");
    mlir::Value next = op.getNext();
    mlir::Value clk = op.getClk();
    mlir::Value data = op.getData();
    mlir::Value reset = op.getReset();
    mlir::Value resetValue = op.getResetValue();
    circuit->performCompReg(next, clk, data, reset, resetValue);
    return mlir::success();
  }

  mlir::LogicalResult visitSeqOp(circt::seq::FromClockOp &op,
                                 Solver::Circuit *circuit) {
    mlir::Value result = op.getResult();
    mlir::Value input = op.getInput();
    // circuit->performCompReg(next, clk, data, reset, resetValue);
    circuit->performFromClock(result, input);
    return mlir::success();
  }
};
} // namespace

LogicalResult LogicExporter::run(ModuleOp &builtinModule) {
  return Visitor(circuit).visit(builtinModule, moduleName);
}

LogicalResult LogicExporter::run(hw::HWModuleOp &module) {
  return Visitor(circuit).visit(module);
}
