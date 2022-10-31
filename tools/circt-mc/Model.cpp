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


mlir::LogicalResult Model::constructModel(mlir::OwningOpRef<mlir::ModuleOp>) {
	return mlir::failure();
}	

	