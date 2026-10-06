//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the types for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4TYPES_TD
#define CIRCT_DIALECT_AXI4_AXI4TYPES_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class AXI4TypeDef<string name, list<Trait> traits = []> :
    TypeDef<AXI4Dialect, name, traits>;

def PortType : AXI4TypeDef<"Port"> {
  let mnemonic = "port";
  let summary = "an AXI4 port interface";
  let description = [{
    The primary interface passed between endpoints and interconnect primitives.
    Carries the address, data, write/read ID and user widths, the access windows
    the port supports, and the maximum writes and reads in flight with any one
    ID, e.g.
    `!axi4.port<addr_width = 32, data_width = 64, write_id_width = 4,
    read_id_width = 4, user_width = 0, windows = <<base = 0x0, last = 0xfff,
    burst_specs = <<fixed, len = 4>>>>, concurrent_writes_per_id = 4,
    concurrent_reads_per_id = 4>`.

    The fields may be given in any order. AXI4 permits address widths up to 64
    bits, data widths that are a power of two between 8 and 1024, and ID widths
    up to 32 bits.
  }];

  let parameters = (ins "uint32_t":$addr_width,
                        "uint32_t":$data_width,
                        "uint32_t":$write_id_width,
                        "uint32_t":$read_id_width,
                        "uint32_t":$user_width,
                        "::circt::axi4::WindowSetAttr":$windows,
                        "uint32_t":$concurrent_writes_per_id,
                        "uint32_t":$concurrent_reads_per_id);

  let assemblyFormat = "`<` struct(params) `>`";

  let genVerifyDecl = 1;
}

def DummiesPortType : AXI4TypeDef<"DummiesPort"> {
  let mnemonic = "dummies.port";
  let summary = "an unparameterised AXI4 port interface";
  let description = [{
    The interface passed between the endpoints and interconnect of the dummies
    subdialect. Widths are inferred later at lowering.
  }];
}

def DummiesManagerAccessType : AXI4TypeDef<"DummiesManagerAccess"> {
  let mnemonic = "dummies.manager_access";
  let summary = "a handle on a dummies manager's accesses";
  let description = [{
    Produced by a dummies manager and consumed by `axi4.dummies.accesses` to
    describe which subordinates the manager reaches.
  }];
}

def DummiesSubordinateAccessType : AXI4TypeDef<"DummiesSubordinateAccess"> {
  let mnemonic = "dummies.subordinate_access";
  let summary = "a handle on a dummies subordinate's accesses";
  let description = [{
    Produced by a dummies subordinate and consumed by `axi4.dummies.accesses`
    to describe which managers reach the subordinate.
  }];
}

#endif // CIRCT_DIALECT_AXI4_AXI4TYPES_TD
