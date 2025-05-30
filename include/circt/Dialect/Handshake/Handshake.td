//===- Handshake.td - Handshake dialect definition ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Handshake dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HANDSHAKE_TD
#define HANDSHAKE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "circt/Dialect/Handshake/HandshakeInterfaces.td"

def Handshake_Dialect : Dialect {
  let name = "handshake";
  let cppNamespace = "::circt::handshake";

  let summary = "Types and operations for the handshake dialect";
  let description = [{
    This dialect defined the `handshake` dialect, modeling dataflow circuits.
    Handshake/dataflow IR is describes independent, unsynchronized processes
    communicating data through First-in First-out (FIFO) communication channels.
  }];

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "circt::esi::ESIDialect",
    "circt::seq::SeqDialect"
  ];
}

include "circt/Dialect/Handshake/HandshakeOps.td"

#endif // HANDSHAKE_TD
