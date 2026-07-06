//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the attributes for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD
#define CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def BurstKindFixed : I32EnumAttrCase<"Fixed", 0, "fixed">;
def BurstKindIncr  : I32EnumAttrCase<"Incr", 1, "incr">;
def BurstKindWrap  : I32EnumAttrCase<"Wrap", 2, "wrap">;

def BurstKind : I32EnumAttr<"BurstKind", "AXI4 burst kind",
    [BurstKindFixed, BurstKindIncr, BurstKindWrap]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::axi4";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class AXI4AttrDef<string name, list<Trait> traits = []> :
    AttrDef<AXI4Dialect, name, traits>;

def BurstSpecAttr : AXI4AttrDef<"BurstSpec"> {
  let mnemonic = "burst_spec";
  let summary = "Describes the burst capabilities of an endpoint";
  let description = [{
    Consists of a burst kind and an increment size, given when the burst kind
    is `incr` or `wrap`, e.g. `#axi4.burst_spec<fixed>` or
    `#axi4.burst_spec<incr, len = 256>`.
  }];

  let parameters = (ins EnumParameter<BurstKind>:$kind,
                        OptionalParameter<"std::optional<uint32_t>">:$len);

  let assemblyFormat = "`<` $kind (`,` `len` `=` $len^)? `>`";

  let genVerifyDecl = 1;
}

def WindowAttr : AXI4AttrDef<"Window"> {
  let mnemonic = "window";
  let summary = "Describes an access window supported by an endpoint";
  let description = [{
    Consists of a base address, a size, and an array of supported burst specs,
    e.g.
    `#axi4.window<base = 16384, size = 256, burst_specs = [#axi4.burst_spec<fixed>]>`.
  }];

  let parameters = (ins "uint64_t":$base,
                        "uint64_t":$size,
                        ArrayRefParameter<"::circt::axi4::BurstSpecAttr">:$burstSpecs);

  let assemblyFormat = [{
    `<` `base` `=` $base `,` `size` `=` $size
        `,` `burst_specs` `=` `[` $burstSpecs `]` `>`
  }];
}

#endif // CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD
