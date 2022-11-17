//===-- Model.h - BMC Model -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the model used in circt-mc.
///
//===----------------------------------------------------------------------===//

#include "Model.h"
#include <z3++.h>

Model::Model(z3::context *c) { z3Context = c; }

mlir::LogicalResult
Model::constructModel(mlir::OwningOpRef<mlir::ModuleOp> *mod) {
  return mlir::failure();
}

void Model::setInitialState() { return; }

void Model::loadCircuitConstraints(z3::solver *s) {
  z3::expr x = z3Context->bool_const("x");
  s->add(x);
}

void Model::loadStateConstraints(z3::solver *s) {
  z3::expr x = z3Context->bool_const("x");
  s->add(x);
}

void Model::runClockCycle() { return; }

void Model::updateInputs() { return; }