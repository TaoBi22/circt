//===-- Model.h - BMC Model Interface ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines an interface for the model used in circt-mc.
///
//===----------------------------------------------------------------------===//

#ifndef MC_MODEL_H
#define MC_MODEL_H

#include "mlir/IR/BuiltinOps.h"

class Model {
	public:
		mlir::LogicalResult constructModel(mlir::OwningOpRef<mlir::ModuleOp>);

	
};

#endif // MC_MODEL_H
