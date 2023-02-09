//===-- Circuit.cpp - intermediate representation for circuits --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines an intermediate representation for circuits acting as
/// an abstraction for constraints defined over an SMT's solver context.
///
//===----------------------------------------------------------------------===//

#include "circt/Verification/Circuit.h"
#include "circt/Verification/LogicExporter.h"
#include "circt/Verification/Solver.h"
#include "circt/Verification/Utility.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "circuit"

/// Add an input to the circuit; internally a new value gets allocated.
void Solver::Circuit::addInput(mlir::Value value) {
  LLVM_DEBUG(lec::dbgs << name << " addInput\n");
  INDENT();
  z3::expr input = allocateValue(value);
  inputs.insert(inputs.end(), input);
  inputsByVal.insert(inputsByVal.end(), value);
}

/// Add an output to the circuit.
void Solver::Circuit::addOutput(mlir::Value value) {
  LLVM_DEBUG(lec::dbgs << name << " addOutput\n");
  // Referenced value already assigned, fetching from expression table.
  z3::expr output = fetchExpr(value);
  outputs.insert(outputs.end(), output);
  outputsByVal.insert(outputsByVal.end(), value);
}

/// Add a clock to the list of clocks.
void Solver::Circuit::addClk(mlir::Value value) {
  if (clks.size() == 1) {
    assert(clks[0] == value && "More than one clock detected - currently "
                               "circt-mc only supports one clock in designs.");
  } else {
    assert(clks.size() == 0 && "Too many clocks added to circuit model.");
    // Check that value is in inputs (i.e. is an external signal and won't be
    // affected by design components)
    auto inputSearch = std::find(inputsByVal.begin(), inputsByVal.end(), value);
    assert(inputSearch != inputsByVal.end() &&
           "Clock is not an input signal - circt-mc currently only supports "
           "external clocks.");
    clks.push_back(value);
  }
}

/// Recover the inputs.
llvm::ArrayRef<z3::expr> Solver::Circuit::getInputs() { return inputs; }

/// Recover the outputs.
llvm::ArrayRef<z3::expr> Solver::Circuit::getOutputs() { return outputs; }

//===----------------------------------------------------------------------===//
// `hw` dialect operations
//===----------------------------------------------------------------------===//

void Solver::Circuit::addConstant(mlir::Value opResult,
                                  const mlir::APInt &opValue) {
  LLVM_DEBUG(lec::dbgs << name << " addConstant\n");
  INDENT();
  allocateConstant(opResult, opValue);
}

void Solver::Circuit::addInstance(llvm::StringRef instanceName,
                                  circt::hw::HWModuleOp op,
                                  mlir::OperandRange arguments,
                                  mlir::ResultRange results) {
  LLVM_DEBUG(lec::dbgs << name << " addInstance\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << "instance name: " << instanceName << "\n");
  LLVM_DEBUG(lec::dbgs << "module name: " << op->getName() << "\n");
  // There is no preventing multiple instances holding the same name.
  // As an hack, a suffix is used to differentiate them.
  std::string suffix = "_" + std::to_string(assignments);
  Circuit instance(name + "@" + instanceName + suffix, solver);
  // Export logic to the instance's circuit by visiting the IR of the
  // instanced module.
  auto res = LogicExporter::Visitor::visitHW(op, &instance);
  assert(res.succeeded() && "Instance visit failed");

  // Constrain the inputs and outputs of the instanced circuit to, respectively,
  // the arguments and results of the instance operation.
  {
    LLVM_DEBUG(lec::dbgs << "instance inputs:\n");
    INDENT();
    auto *input = instance.inputs.begin();
    for (mlir::Value argument : arguments) {
      LLVM_DEBUG(lec::dbgs << "input\n");
      z3::expr argExpr = fetchExpr(argument);
      solver->solver.add(argExpr == *input++);
    }
  }
  {
    LLVM_DEBUG(lec::dbgs << "instance results:\n");
    INDENT();
    auto *output = instance.outputs.begin();
    for (circt::OpResult result : results) {
      z3::expr resultExpr = allocateValue(result);
      solver->solver.add(resultExpr == *output++);
    }
  }
}

//===----------------------------------------------------------------------===//
// `comb` dialect operations
//===----------------------------------------------------------------------===//

// This macro implements the perform function for a `comb` operation accepting
// a variadic number of operands.
#define performVariadicCombOp(OP_NAME, Z3_OPERATION)                           \
  void Solver::Circuit::perform##OP_NAME(mlir::Value result,                   \
                                         mlir::OperandRange operands) {        \
    LLVM_DEBUG(lec::dbgs << name << " perform" #OP_NAME "\n");                 \
    INDENT();                                                                  \
    combTransformTable.insert(                                                 \
        std::pair(result, std::pair(operands, [](auto op1, auto op2) {         \
                    return Z3_OPERATION;                                       \
                  })));                                                        \
    wires.insert(wires.end(), result);                                         \
    variadicOperation(result, operands,                                        \
                      [](auto op1, auto op2) { return Z3_OPERATION; });        \
  }

// This macro implements the perform function for a `comb` operation accepting
// two operands.
#define performBinaryCombOp(OP_NAME, Z3_OPERATION)                             \
  void Solver::Circuit::perform##OP_NAME(mlir::Value result, mlir::Value lhs,  \
                                         mlir::Value rhs) {                    \
    LLVM_DEBUG(lec::dbgs << name << " perform" #OP_NAME "\n");                 \
    INDENT();                                                                  \
    LLVM_DEBUG(lec::dbgs << "lhs:\n");                                         \
    z3::expr lhsExpr = fetchExpr(lhs);                                         \
    LLVM_DEBUG(lec::dbgs << "rhs:\n");                                         \
    z3::expr rhsExpr = fetchExpr(rhs);                                         \
    z3::expr op = z3::Z3_OPERATION(lhsExpr, rhsExpr);                          \
    combTransformTable.insert(std::pair(                                       \
        result, std::pair(std::tuple(lhs, rhs), [](auto op1, auto op2) {       \
          return z3::Z3_OPERATION(op1, op2);                                   \
        })));                                                                  \
    constrainResult(result, op);                                               \
  }

performVariadicCombOp(Add, op1 + op2);

performVariadicCombOp(And, z3::operator&(op1, op2));

performVariadicCombOp(Concat, z3::concat(op1, op2));

performBinaryCombOp(DivS, operator/);

performBinaryCombOp(DivU, udiv);

void Solver::Circuit::performExtract(mlir::Value result, mlir::Value input,
                                     uint32_t lowBit) {
  LLVM_DEBUG(lec::dbgs << name << " performExtract\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << "input:\n");
  z3::expr inputExpr = fetchExpr(input);
  unsigned width = result.getType().getIntOrFloatBitWidth();
  LLVM_DEBUG(lec::dbgs << "width: " << width << "\n");
  z3::expr extract = inputExpr.extract(lowBit + width - 1, lowBit);
  combTransformTable.insert(std::pair(
      result, std::pair(std::make_tuple(input), [lowBit, width](auto op1) {
        return op1.extract(lowBit + width - 1, lowBit);
      })));
  constrainResult(result, extract);
}

void Solver::Circuit::performICmp(mlir::Value result,
                                  circt::comb::ICmpPredicate predicate,
                                  mlir::Value lhs, mlir::Value rhs) {
  LLVM_DEBUG(lec::dbgs << name << " performICmp\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << "lhs:\n");
  z3::expr lhsExpr = fetchExpr(lhs);
  LLVM_DEBUG(lec::dbgs << "rhs:\n");
  z3::expr rhsExpr = fetchExpr(rhs);
  z3::expr icmp(solver->context);

  switch (predicate) {
  case circt::comb::ICmpPredicate::eq:
  // Multi-valued logic is not accounted for.
  case circt::comb::ICmpPredicate::ceq:
  case circt::comb::ICmpPredicate::weq:
    icmp = boolToBv(lhsExpr == rhsExpr);
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(op1 == op2);
        })));
    break;
  case circt::comb::ICmpPredicate::ne:
  // Multi-valued logic is not accounted for.
  case circt::comb::ICmpPredicate::cne:
  case circt::comb::ICmpPredicate::wne:
    icmp = boolToBv(lhsExpr != rhsExpr);
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(op1 != op2);
        })));
    break;
  case circt::comb::ICmpPredicate::slt:
    icmp = boolToBv(z3::slt(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::slt(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::sle:
    icmp = boolToBv(z3::sle(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::sle(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::sgt:
    icmp = boolToBv(z3::sgt(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::sgt(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::sge:
    icmp = boolToBv(z3::sge(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::sge(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::ult:
    icmp = boolToBv(z3::ult(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::ult(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::ule:
    icmp = boolToBv(z3::ule(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::ule(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::ugt:
    icmp = boolToBv(z3::ugt(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::ugt(op1, op2));
        })));
    break;
  case circt::comb::ICmpPredicate::uge:
    icmp = boolToBv(z3::uge(lhsExpr, rhsExpr));
    combTransformTable.insert(std::pair(
        result, std::pair(std::tuple(lhs, rhs), [this](auto op1, auto op2) {
          return boolToBv(z3::uge(op1, op2));
        })));
    break;
  };

  constrainResult(result, icmp);
}

performBinaryCombOp(ModS, smod);

performBinaryCombOp(ModU, urem);

performVariadicCombOp(Mul, op1 *op2);

void Solver::Circuit::performMux(mlir::Value result, mlir::Value cond,
                                 mlir::Value trueValue,
                                 mlir::Value falseValue) {
  LLVM_DEBUG(lec::dbgs << name << " performMux\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << "cond:\n");
  z3::expr condExpr = fetchExpr(cond);
  LLVM_DEBUG(lec::dbgs << "trueValue:\n");
  z3::expr tvalue = fetchExpr(trueValue);
  LLVM_DEBUG(lec::dbgs << "falseValue:\n");
  z3::expr fvalue = fetchExpr(falseValue);
  // Conversion due to z3::ite requiring a bool rather than a bitvector.
  z3::expr mux = z3::ite(bvToBool(condExpr), tvalue, fvalue);
  combTransformTable.insert(
      std::pair(result, std::pair(std::make_tuple(cond, trueValue, falseValue),
                                  [this](auto op1, auto op2, auto op3) {
                                    return z3::ite(bvToBool(op1), op2, op3);
                                  })));
  constrainResult(result, mux);
}

performVariadicCombOp(Or, op1 | op2);

void Solver::Circuit::performParity(mlir::Value result, mlir::Value input) {
  LLVM_DEBUG(lec::dbgs << name << " performParity\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << "input:\n");
  z3::expr inputExpr = fetchExpr(input);

  unsigned width = inputExpr.get_sort().bv_size();

  // input has 1 or more bits
  z3::expr parity = inputExpr.extract(0, 0);
  // calculate parity with every other bit
  for (unsigned int i = 1; i < width; i++) {
    parity = parity ^ inputExpr.extract(i, i);
  }

  combTransformTable.insert(
      std::pair(result, std::pair(std::make_tuple(input), [width](auto op1) {
                  z3::expr parity = op1.extract(0, 0);
                  // calculate parity with every other bit
                  for (unsigned int i = 1; i < width; i++) {
                    parity = parity ^ op1.extract(i, i);
                  }
                  return parity;
                })));

  constrainResult(result, parity);
}

void Solver::Circuit::performReplicate(mlir::Value result, mlir::Value input) {
  LLVM_DEBUG(lec::dbgs << name << " performReplicate\n");
  INDENT();
  LLVM_DEBUG(lec::dbgs << "input:\n");
  z3::expr inputExpr = fetchExpr(input);

  unsigned int final = result.getType().getIntOrFloatBitWidth();
  unsigned int initial = input.getType().getIntOrFloatBitWidth();
  unsigned int times = final / initial;
  LLVM_DEBUG(lec::dbgs << "replies: " << times << "\n");

  z3::expr replicate = inputExpr;
  for (unsigned int i = 1; i < times; i++) {
    replicate = z3::concat(replicate, inputExpr);
  }

  combTransformTable.insert(
      std::pair(result, std::pair(std::make_tuple(input), [times](auto op1) {
                  z3::expr replicate = op1;
                  for (unsigned int i = 1; i < times; i++) {
                    replicate = z3::concat(replicate, op1);
                  }
                  return replicate;
                })));

  constrainResult(result, replicate);
}

performBinaryCombOp(Shl, shl);

// Arithmetic shift right.
performBinaryCombOp(ShrS, ashr);

// Logical shift right.
performBinaryCombOp(ShrU, lshr);

performVariadicCombOp(Sub, op1 - op2);

performVariadicCombOp(Xor, op1 ^ op2);

/// Helper function for performing a variadic operation: it executes a lambda
/// over a range of operands.
void Solver::Circuit::variadicOperation(
    mlir::Value result, mlir::OperandRange operands,
    llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)>
        operation) {
  LLVM_DEBUG(lec::dbgs << "variadic operation\n");
  INDENT();
  // Vacuous base case.
  auto it = operands.begin();
  mlir::Value operand = *it;
  z3::expr varOp = exprTable.find(operand)->second;
  {
    LLVM_DEBUG(lec::dbgs << "first operand:\n");
    INDENT();
    LLVM_DEBUG(lec::printValue(operand));
  }
  ++it;
  // Inductive step.
  while (it != operands.end()) {
    operand = *it;
    varOp = operation(varOp, exprTable.find(operand)->second);
    {
      LLVM_DEBUG(lec::dbgs << "next operand:\n");
      INDENT();
      LLVM_DEBUG(lec::printValue(operand));
    }
    ++it;
  };
  constrainResult(result, varOp);
}

/// Allocates an IR value in the logical backend and returns its representing
/// expression.
z3::expr Solver::Circuit::allocateValue(mlir::Value value) {
  std::string valueName = name + "%" + std::to_string(assignments++);
  auto nameInsertion = nameTable.insert(std::pair(value, valueName));
  assert(nameInsertion.second && "Name not inserted in state table");
  LLVM_DEBUG(lec::dbgs << "allocating value:\n");
  INDENT();
  mlir::Type type = value.getType();
  assert(type.isSignlessInteger() && "Unsupported type");
  unsigned int width = type.getIntOrFloatBitWidth();
  // Technically allowed for the `hw` dialect but
  // disallowed for `comb` operations; should check separately.
  assert(width > 0 && "0-width integers are not supported"); // NOLINT
  z3::expr expr = solver->context.bv_const(valueName.c_str(), width);
  LLVM_DEBUG(lec::printExpr(expr));
  LLVM_DEBUG(lec::printValue(value));
  auto exprInsertion = exprTable.insert(std::pair(value, expr));
  assert(exprInsertion.second && "Value not inserted in expression table");
  // Populate state table
  std::string stateName = valueName + std::string("_init");
  z3::expr stateExpr = solver->context.bv_const(stateName.c_str(), width);
  auto stateInsertion = stateTable.insert(std::pair(value, stateExpr));
  assert(stateInsertion.second && "Value not inserted in state table");
  mlir::Builder builder(solver->mlirCtx);
  mlir::StringAttr symbol = builder.getStringAttr(valueName);
  auto symInsertion = solver->symbolTable.insert(std::pair(symbol, value));
  assert(symInsertion.second && "Value not inserted in symbol table");
  mlir::StringAttr stateSymbol = builder.getStringAttr(stateName);
  auto symStateInsertion =
      solver->symbolTable.insert(std::pair(stateSymbol, value));
  assert(symStateInsertion.second && "State not inserted in symbol table");
  return expr;
}

/// Allocates a constant value in the logical backend and returns its
/// representing expression.
void Solver::Circuit::allocateConstant(mlir::Value result,
                                       const mlir::APInt &value) {
  // `The constant operation produces a constant value
  //  of standard integer type without a sign`
  const z3::expr constant =
      solver->context.bv_val(value.getZExtValue(), value.getBitWidth());
  auto insertion = exprTable.insert(std::pair(result, constant));
  assert(insertion.second && "Constant not inserted in expression table");
  auto stateInsertion = stateTable.insert(std::pair(result, constant));
  assert(stateInsertion.second && "Value not inserted in state table");
  LLVM_DEBUG(lec::printExpr(constant));
  LLVM_DEBUG(lec::printValue(result));
}

/// Fetches the corresponding logical expression for a given IR value.
z3::expr Solver::Circuit::fetchExpr(mlir::Value &value) {
  z3::expr expr = exprTable.find(value)->second;
  INDENT();
  LLVM_DEBUG(lec::printExpr(expr));
  LLVM_DEBUG(lec::printValue(value));
  return expr;
}

/// Constrains the result of a MLIR operation to be equal a given logical
/// express, simulating an assignment.
void Solver::Circuit::constrainResult(mlir::Value &result, z3::expr &expr) {
  LLVM_DEBUG(lec::dbgs << "constraining result:\n");
  INDENT();
  {
    LLVM_DEBUG(lec::dbgs << "result expression:\n");
    INDENT();
    LLVM_DEBUG(lec::printExpr(expr));
  }
  z3::expr resExpr = allocateValue(result);
  z3::expr constraint = resExpr == expr;
  {
    LLVM_DEBUG(lec::dbgs << "adding constraint:\n");
    INDENT();
    LLVM_DEBUG(lec::dbgs << constraint.to_string() << "\n");
  }
  solver->solver.add(constraint);
}

/// Convert from bitvector to bool sort.
z3::expr Solver::Circuit::bvToBool(const z3::expr &condition) {
  // bitvector is true if it's different from 0
  return condition != 0;
}

/// Convert from a boolean sort to the corresponding 1-width bitvector.
z3::expr Solver::Circuit::boolToBv(const z3::expr &condition) {
  return z3::ite(condition, solver->context.bv_val(1, 1),
                 solver->context.bv_val(0, 1));
}

/// Push solver constraints assigning registers and inputs to their current
/// state
void Solver::Circuit::loadStateConstraints() {
  for (auto input : inputsByVal) {
    auto symbolPair = exprTable.find(input);
    assert(symbolPair != exprTable.end() &&
           "Z3 expression not found for input value");
    auto statePair = stateTable.find(input);
    assert(statePair != stateTable.end() &&
           "Z3 state not found for input value");
    solver->solver.add(symbolPair->second == statePair->second);
  }
  for (auto reg : regs) {
    mlir::Value regData;
    if (auto *compReg = std::get_if<CompRegStruct>(&reg)) {
      regData = compReg->data;
    } else if (auto *firReg = std::get_if<FirRegStruct>(&reg)) {
      regData = firReg->data;
    }
    auto symbolPair = exprTable.find(regData);
    assert(symbolPair != exprTable.end() &&
           "Z3 expression not found for register output");
    auto statePair = stateTable.find(regData);
    assert(statePair != stateTable.end() &&
           "Z3 state not found for register output");

    solver->solver.add(symbolPair->second == statePair->second);
  }
  // Combinatorial values are handled by the constraints we already have, so we
  // do not need their state
  return;
}

/// Execute a clock posedge (i.e. update registers and combinatorial logic)
void Solver::Circuit::runClockPosedge() {
  for (auto clk : clks) {
    // Currently we explicitly handle only one clock, so we can just update
    // every clock in clks (of which there are 0 or 1)
    stateTable.find(clk)->second = solver->context.bv_val(1, 1);
  }
  for (auto reg : regs) {
    // Fetch values from reg structs
    mlir::Value input;
    mlir::Value data;
    mlir::Value reset;
    mlir::Value resetValue;
    if (auto *compReg = std::get_if<CompRegStruct>(&reg)) {
      input = compReg->input;
      data = compReg->data;
      reset = compReg->reset;
      resetValue = compReg->resetValue;
    } else if (auto *firReg = std::get_if<FirRegStruct>(&reg)) {
      input = firReg->next;
      data = firReg->data;
      reset = firReg->reset;
      resetValue = firReg->resetValue;
    }
    // Currently, there is no difference in CompReg and FirReg handling, as
    // async resets aren't supported
    z3::expr inputState = stateTable.find(input)->second;
    // Make sure that a reset value is present
    if (reset) {
      z3::expr resetState = stateTable.find(reset)->second;
      z3::expr resetValueState = stateTable.find(resetValue)->second;
      z3::expr newState =
          z3::ite(bvToBool(resetState), resetValueState, inputState);
      stateTable.find(data)->second = newState;
    } else {
      // Otherwise, simply update output state to be the same as input state
      stateTable.find(data)->second = inputState;
    }
  }
  // Update combinational updates so register outputs can propagate
  applyCombUpdates();
  return;
}

/// Execute a clock negedge (i.e. update combinatorial logic)
void Solver::Circuit::runClockNegedge() {
  for (auto clk : clks) {
    // Currently we explicitly handle only one clock, so we can just update
    // every clock in clks (of which there are 0 or 1)
    stateTable.find(clk)->second = solver->context.bv_val(0, 1);
  }
  // Update combinational updates so changes in inputs can propagate
  applyCombUpdates();
  return;
}

/// Assign a new set of symbolic values to all inputs
void Solver::Circuit::updateInputs(int count, bool posedge) {
  mlir::Builder builder(solver->mlirCtx);
  for (auto input : inputsByVal) {
    // We update clocks literally, so skip this for clocks
    if (std::find(clks.begin(), clks.end(), input) != clks.end()) {
      continue;
    }
    llvm::DenseMap<mlir::Value, z3::expr>::iterator currentStatePair =
        stateTable.find(input);
    if (currentStatePair != stateTable.end()) {
      int width = input.getType().getIntOrFloatBitWidth();
      std::string valueName = nameTable.find(input)->second;
      std::string edgeString(posedge ? "_pos" : "_neg");
      std::string symbolName =
          (valueName + "_" + std::to_string(count) + edgeString).c_str();
      currentStatePair->second =
          solver->context.bv_const(symbolName.c_str(), width);
      mlir::StringAttr symbol = builder.getStringAttr(symbolName);
      auto symInsertion = solver->symbolTable.insert(std::pair(symbol, input));
      assert(symInsertion.second && "Value not inserted in symbol table");
    }
  }
  return;
}

/// Check that the properties hold for the current state
bool Solver::Circuit::checkState() {
  solver->solver.push();
  loadStateConstraints();
  auto result = solver->solver.check();
  solver->solver.pop();
  switch (result) {
  case z3::sat:
    solver->printModel();
    return false;
    break;
  case z3::unsat:
    return true;
    break;
  default:
    // TODO: maybe add handler for other return vals?
    return false;
  }
}

/// Execute a clock cycle and check that the properties hold throughout
bool Solver::Circuit::checkCycle(int count) {
  updateInputs(count, true);
  runClockPosedge();
  if (!checkState()) {
    //Print all the solver constraints
    lec::dbgs() << "Solver constraints:\n";
    for (auto constraint : solver->solver.assertions()) {
      	// Convert the assertion to a string
	llvm::errs() << constraint.to_string() << "\n";
    }
    return false;
  }
  updateInputs(count, false);
  runClockNegedge();
  if (!checkState()) {
    return false;
  }
  return true;
}

/// Update combinatorial logic states (to propagate new inputs/reg outputs)
void Solver::Circuit::applyCombUpdates() {
  for (auto wire : wires) {
    auto wireTransformPair = combTransformTable.find(wire);
    assert(wireTransformPair != combTransformTable.end() &&
           "Combinational value to update has no update function");
    auto wireTransform = wireTransformPair->second;
    if (auto *transform = std::get_if<std::pair<
            mlir::OperandRange,
            llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)>>>(
            &wireTransform)) {
      applyCombVariadicOperation(wire, *transform);
    } else if (auto *transform = std::get_if<
                   std::pair<std::tuple<mlir::Value>,
                             llvm::function_ref<z3::expr(const z3::expr &)>>>(
                   &wireTransform)) {
      mlir::Value operand = std::get<0>(transform->first);
      llvm::function_ref<z3::expr(const z3::expr &)> transformFunc =
          transform->second;
      z3::expr operandExpr = stateTable.find(operand)->second;
      stateTable.find(wire)->second = transformFunc(operandExpr);
    } else if (auto *transform = std::get_if<
                   std::pair<std::tuple<mlir::Value, mlir::Value>,
                             llvm::function_ref<z3::expr(const z3::expr &,
                                                         const z3::expr &)>>>(
                   &wireTransform)) {
      mlir::Value firstOperand = std::get<0>(transform->first);
      mlir::Value secondOperand = std::get<1>(transform->first);
      llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)>
          transformFunc = transform->second;
      z3::expr firstOperandExpr = stateTable.find(firstOperand)->second;
      z3::expr secondOperandExpr = stateTable.find(secondOperand)->second;
      stateTable.find(wire)->second =
          transformFunc(firstOperandExpr, secondOperandExpr);
    } else if (auto *transform = std::get_if<std::pair<
                   std::tuple<mlir::Value, mlir::Value, mlir::Value>,
                   llvm::function_ref<z3::expr(
                       const z3::expr &, const z3::expr &, const z3::expr &)>>>(
                   &wireTransform)) {
      mlir::Value firstOperand = std::get<0>(transform->first);
      mlir::Value secondOperand = std::get<1>(transform->first);
      mlir::Value thirdOperand = std::get<2>(transform->first);
      llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &,
                                  const z3::expr &)>
          transformFunc = transform->second;
      z3::expr firstOperandExpr = stateTable.find(firstOperand)->second;
      z3::expr secondOperandExpr = stateTable.find(secondOperand)->second;
      z3::expr thirdOperandExpr = stateTable.find(thirdOperand)->second;
      stateTable.find(wire)->second =
          transformFunc(firstOperandExpr, secondOperandExpr, thirdOperandExpr);
    }
  }
}

/// Helper function for applying a variadic update operation: it executes a
/// lambda over a range of operands and updates the state.
void Solver::Circuit::applyCombVariadicOperation(
    mlir::Value result,
    std::pair<mlir::OperandRange,
              llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)>>
        operationPair) {
  LLVM_DEBUG(lec::dbgs << "comb variadic operation\n");
  INDENT();
  mlir::OperandRange operands = operationPair.first;
  llvm::function_ref<z3::expr(const z3::expr &, const z3::expr &)> operation =
      operationPair.second;
  // Vacuous base case.
  auto it = operands.begin();
  mlir::Value operand = *it;
  z3::expr varOp = exprTable.find(operand)->second;
  {
    LLVM_DEBUG(lec::dbgs << "first operand:\n");
    INDENT();
    LLVM_DEBUG(lec::printValue(operand));
  }
  ++it;
  // Inductive step.
  while (it != operands.end()) {
    operand = *it;
    varOp = operation(varOp, exprTable.find(operand)->second);
    {
      LLVM_DEBUG(lec::dbgs << "next operand:\n");
      INDENT();
      LLVM_DEBUG(lec::printValue(operand));
    }
    ++it;
  };
  stateTable.find(result)->second = varOp;
}

//===----------------------------------------------------------------------===//
// `seq` dialect operations
//===----------------------------------------------------------------------===//
void Solver::Circuit::performCompReg(mlir::Value input, mlir::Value clk,
                                     mlir::Value data, mlir::Value reset,
                                     mlir::Value resetValue) {
  z3::expr regData = allocateValue(data);
  CompRegStruct reg;
  reg.input = input;
  reg.clk = clk;
  reg.data = data;
  reg.reset = reset;
  reg.resetValue = resetValue;
  regs.insert(regs.end(), reg);
  addClk(clk);
  // TODO THIS IS TEMPORARY FOR TESTING
  z3::expr inExpr = exprTable.find(input)->second;
  z3::expr outExpr = exprTable.find(data)->second;
  auto rstPair = exprTable.find(reset);
  z3::expr clkExpr = exprTable.find(clk)->second;
  LLVM_DEBUG(lec::dbgs << "Input: " << nameTable.find(input)->second << "\n");
  LLVM_DEBUG(lec::dbgs << "Output: " << nameTable.find(data)->second << "\n");
  if (rstPair != exprTable.end())
    solver->solver.add(!z3::implies(
        bvToBool(clkExpr) && !bvToBool(rstPair->second), inExpr == outExpr));
}

void Solver::Circuit::performFirReg(mlir::Value next, mlir::Value clk,
                                    mlir::Value data, mlir::Value reset,
                                    mlir::Value resetValue) {
  z3::expr regData = allocateValue(data);
  FirRegStruct reg;
  reg.next = next;
  reg.clk = clk;
  reg.data = data;
  reg.reset = reset;
  reg.resetValue = resetValue;
  regs.insert(regs.end(), reg);
  clks.insert(clks.end(), clk);
  // TODO THIS IS TEMPORARY FOR TESTING
  z3::expr inExpr = exprTable.find(next)->second;
  z3::expr outExpr = exprTable.find(data)->second;
  auto rstPair = exprTable.find(reset);
  z3::expr clkExpr = exprTable.find(clk)->second;
  LLVM_DEBUG(lec::dbgs << "Input: " << nameTable.find(next)->second << "\n");
  LLVM_DEBUG(lec::dbgs << "Output: " << nameTable.find(data)->second << "\n");
  if (rstPair != exprTable.end())
    solver->solver.add(!z3::implies(
        bvToBool(clkExpr) && !bvToBool(rstPair->second), inExpr == outExpr));
}

#undef DEBUG_TYPE
