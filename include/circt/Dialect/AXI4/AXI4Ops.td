//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_TD
#define CIRCT_DIALECT_AXI4_AXI4OPS_TD

include "circt/Dialect/AXI4/AXI4Attributes.td"
include "circt/Dialect/AXI4/AXI4Dialect.td"
include "circt/Dialect/AXI4/AXI4Types.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operations in this dialect.
class AXI4Op<string mnemonic, list<Trait> traits = []> :
    Op<AXI4Dialect, mnemonic, traits>;

// Constrains each `!axi4.port` result to at most one use
def PortResultsAtMostOneUse : NativeOpTrait<"PortResultsAtMostOneUse"> {
  let cppNamespace = "::circt::axi4::OpTrait";
}

// Constrain an argument to the hw representation of a given AXI channel
class PayloadMatches<string channel, string arg> : TypesMatchWith<
    channel # " payload matches the port type", "port", arg,
    "::circt::axi4::getChannelPayloadType("
      "::llvm::cast<::circt::axi4::PortType>($_self), "
      "::circt::axi4::AXI4Channel::" # channel # ")">;

// Constrain an argument to an integer as wide as a field of a given port type
class PortWidthMatches<string arg, string description, string width> :
    TypesMatchWith<description, "port", arg,
    "::mlir::IntegerType::get($_self.getContext(), "
      "::llvm::cast<::circt::axi4::PortType>($_self)." # width # ")">;

// Ordered list of AXI4 channels
defvar axi4Channels = ["AW", "W", "B", "AR", "R"];

// Base class for the ops bridging an !axi4.port and its hw channel structs,
// with automatic struct type enforcement
class AXI4ChannelStructOp<string mnemonic, list<Trait> traits = []> :
    AXI4Op<mnemonic, !listconcat(traits,
        !foreach(c, axi4Channels, PayloadMatches<c, !tolower(c)>))>;

// Base class for the operations in the dummies subdialect.
class AXI4DummiesOp<string mnemonic, list<Trait> traits = []> :
    AXI4Op<"dummies." # mnemonic, traits>;

def AbstractManagerOp : AXI4Op<"abstract_manager", [PortResultsAtMostOneUse]> {
  let summary = "A stand-alone AXI4 manager endpoint";
  let description = [{
    Models a manager endpoint with no concrete implementation, for abstract
    network modelling. Returns the `!axi4.port` it drives.

    Example:
    ```mlir
    %mgr = axi4.abstract_manager %clk, %rst_ni : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset);
  let results = (outs PortType:$port);

  let assemblyFormat =
      "$clock `,` $reset attr-dict `:` qualified(type($port))";
}

def AbstractSubordinateOp : AXI4Op<"abstract_subordinate"> {
  let summary = "A stand-alone AXI4 subordinate endpoint";
  let description = [{
    Models a subordinate endpoint with no concrete implementation, for abstract
    network modelling. Takes the upstream `!axi4.port` it responds to.

    Example:
    ```mlir
    axi4.abstract_subordinate %clk, %rst_ni, %mgr : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);

  let assemblyFormat =
      "$clock `,` $reset `,` $upstream attr-dict `:` qualified(type($upstream))";
}

def ChannelStructsToPortOp : AXI4ChannelStructOp<"channel_structs_to_port",
    [PortResultsAtMostOneUse]> {
  let summary = "Converts a HW expression of an AXI4 interface to a port";
  let description = [{
    Bridges from a HW dialect expression of an AXI4 interface to an
    `!axi4.port`. Takes the AW, W and AR channel payloads and valids, and the B
    and R readys; returns the port along with the AW, W and AR readys and the B
    and R payloads and valids.

    Example:
    ```mlir
    %port, %aw_ready, %w_ready, %b, %b_valid, %ar_ready, %r, %r_valid =
      axi4.channel_structs_to_port %clk, %rst_ni
        aw %aw, %aw_valid w %w, %w_valid b %b_ready
        ar %ar, %ar_valid r %r_ready
      : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset,
                       StructType:$aw, I1:$aw_valid,
                       StructType:$w, I1:$w_valid,
                       I1:$b_ready,
                       StructType:$ar, I1:$ar_valid,
                       I1:$r_ready);
  let results = (outs PortType:$port,
                      I1:$aw_ready, I1:$w_ready,
                      StructType:$b, I1:$b_valid,
                      I1:$ar_ready,
                      StructType:$r, I1:$r_valid);

  let assemblyFormat = [{
    $clock `,` $reset
      `aw` $aw `,` $aw_valid `w` $w `,` $w_valid `b` $b_ready
      `ar` $ar `,` $ar_valid `r` $r_ready
      attr-dict `:` qualified(type($port))
  }];
}

def PortToChannelStructsOp : AXI4ChannelStructOp<"port_to_channel_structs"> {
  let summary = "Converts an AXI4 port to a HW expression of an AXI4 interface";
  let description = [{
    Bridges from an `!axi4.port` to a HW dialect expression of an AXI4
    interface. Takes the port, the AW, W and AR channel readys, and the B and R
    payloads and valids; returns the AW, W and AR payloads and valids, and the
    B and R readys.

    Example:
    ```mlir
    %aw, %aw_valid, %w, %w_valid, %b_ready, %ar, %ar_valid, %r_ready =
      axi4.port_to_channel_structs %clk, %rst_ni, %port
        aw %aw_ready w %w_ready b %b, %b_valid
        ar %ar_ready r %r, %r_valid
      : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$port,
                       I1:$aw_ready, I1:$w_ready,
                       StructType:$b, I1:$b_valid,
                       I1:$ar_ready,
                       StructType:$r, I1:$r_valid);
  let results = (outs StructType:$aw, I1:$aw_valid,
                      StructType:$w, I1:$w_valid,
                      I1:$b_ready,
                      StructType:$ar, I1:$ar_valid,
                      I1:$r_ready);

  let assemblyFormat = [{
    $clock `,` $reset `,` $port
      `aw` $aw_ready `w` $w_ready `b` $b `,` $b_valid
      `ar` $ar_ready `r` $r `,` $r_valid
      attr-dict `:` qualified(type($port))
  }];
}

def XbarOp : AXI4Op<"xbar", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 crossbar interconnect";
  let description = [{
    Routes the requests of a set of upstream managers onto a set of downstream
    subordinates. Every address a manager can issue to must be covered by
    exactly one downstream port, which must support at least the bursts the
    manager issues there.

    Example:
    ```mlir
    %mem, %periph = axi4.xbar %clk, %rst_ni mgrs %core, %debug
      : (!core_mgr, !debug_mgr) -> (!mem_sub, !periph_sub)
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset,
                       Variadic<PortType>:$upstream);
  let results = (outs Variadic<PortType>:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `mgrs` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def CutOp : AXI4Op<"cut", [PortResultsAtMostOneUse,
    TypesMatchWith<"downstream port must match the upstream port", "upstream",
                   "downstream", "$_self">]> {
  let summary = "An AXI4 register cut";
  let description = [{
    Breaks the combinational paths through an AXI4 connection, at the cost of a
    cycle of latency in each direction.

    Example:
    ```mlir
    %downstream = axi4.cut %clk, %rst_ni, %upstream
      : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat =
      "$clock `,` $reset `,` $upstream attr-dict `:` qualified(type($upstream))";
}

def CDCOp : AXI4Op<"cdc", [PortResultsAtMostOneUse,
    TypesMatchWith<"downstream port must match the upstream port", "upstream",
                   "downstream", "$_self">]> {
  let summary = "An AXI4 clock domain crossing";
  let description = [{
    Carries an AXI4 connection from one clock domain into another. Not a reset
    crossing.

    Example:
    ```mlir
    %downstream =
      axi4.cdc from %upstream_clk to %downstream_clk, %rst_ni, %upstream
        : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$upstream_clock,
                       ClockType:$downstream_clock,
                       I1:$reset, PortType:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat = [{
    `from` $upstream_clock `to` $downstream_clock `,` $reset `,` $upstream
      attr-dict `:` qualified(type($upstream))
  }];

  let hasCanonicalizeMethod = 1;
}

def DWConverterOp : AXI4Op<"data_width_converter", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 data width converter";
  let description = [{
    Changes the data width of an AXI4 connection.

    Example:
    ```mlir
    %downstream = axi4.data_width_converter %clk, %rst_ni, %upstream
      : (!axi4.port<...>) -> !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `,` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def IWConverterOp : AXI4Op<"id_width_converter", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 ID width converter";
  let description = [{
    Changes the ID widths of an AXI4 connection.

    Widening prepends zeroes to each ID. Narrowing remaps the IDs in flight
    onto the narrower ones, which orders transactions that end up sharing an
    ID.

    Example:
    ```mlir
    %downstream = axi4.id_width_converter %clk, %rst_ni, %upstream
      : (!axi4.port<...>) -> !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `,` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def BurstSplitterOp : AXI4Op<"burst_splitter", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 burst splitter";
  let description = [{
    Splits every burst on an AXI4 connection into single-beat transactions.

    A single beat has no wrapping to do, so a `wrap` burst splits into `incr`
    beats.

    Example:
    ```mlir
    %downstream = axi4.burst_splitter %clk, %rst_ni, %upstream
      : (!axi4.port<...>) -> !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `,` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def BurstUnwrapperOp : AXI4Op<"burst_unwrapper", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 burst unwrapper";
  let description = [{
    Converts every wrapping burst on an AXI4 connection into up to 2 incremental
    bursts.

    A burst starting on its wrap boundary increments throughout, so it becomes
    one `incr` burst of the same length. One starting anywhere else becomes two,
    split at the wrap boundary.

    Example:
    ```mlir
    %downstream = axi4.burst_unwrapper %clk, %rst_ni, %upstream
      : (!axi4.port<...>) -> !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `,` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def DemuxOp : AXI4Op<"demux", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 demultiplexer";
  let description = [{
    Demultiplexes an AXI4 port into several ports, only one of which has its
    AW/AR valid signals high based on the address mapping inferred from this
    op's return types.

    Example:
    ```mlir
    %downstream1, %downstream2 = axi4.demux %clk, %rst_ni, %upstream
      : (!axi4.port<...>) -> (!axi4.port<...>, !axi4.port<...>)
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$upstream);
  let results = (outs Variadic<PortType>:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `,` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def MuxOp : AXI4Op<"mux", [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 multiplexer";
  let description = [{
    Multiplexes between several AXI4 ports. Concurrent requests are arbitrated
    between using round robin arbitration.

    Example:
    ```mlir
    %downstream = axi4.mux %clk, %rst_ni, %upstream1, %upstream2
      : (!axi4.port<...>, !axi4.port<...>) -> (!axi4.port<...>)
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, Variadic<PortType>:$upstream);
  let results = (outs PortType:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `,` $upstream attr-dict `:`
      functional-type($upstream, $downstream)
  }];

  let hasVerifier = 1;
}

def ToMemOp : AXI4Op<"to_mem", [
    PortWidthMatches<"read_data", "read data is as wide as the port's data",
                     "getDataWidth()">,
    PortWidthMatches<"address", "address is as wide as the port's addresses",
                     "getAddrWidth()">,
    PortWidthMatches<"write_data", "write data is as wide as the port's data",
                     "getDataWidth()">,
    PortWidthMatches<"strobe", "strobe has a bit per byte of the port's data",
                     "getDataWidth() / 8">]> {
  let summary = "An AXI4 to memory interface converter";
  let description = [{
    Converts an AXI4 port into a generic memory interface, for building a
    memory subordinate. Takes the port along with the memory's read response
    valid and read data; returns the valid, address, write data, strobe and
    write enable of the requests it drives the memory with.

    The memory is assumed to accept every request immediately.

    Example:
    ```mlir
    %valid, %addr, %wdata, %strb, %we =
      axi4.to_mem %clk, %rst_ni, %port read %rvalid, %rdata
        : !axi4.port<...>
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset, PortType:$port,
                       I1:$read_valid, AnyInteger:$read_data);
  let results = (outs I1:$request_valid, AnyInteger:$address,
                      AnyInteger:$write_data, AnyInteger:$strobe,
                      I1:$write_enable);

  let assemblyFormat = [{
    $clock `,` $reset `,` $port `read` $read_valid `,` $read_data
      attr-dict `:` qualified(type($port))
  }];
}

def DummiesExtManagerOp : AXI4DummiesOp<"ext_manager",
                                        [PortResultsAtMostOneUse]> {
  let summary = "An AXI4 manager outside the module describing the network";
  let description = [{
    Models a manager whose implementation sits outside the module the network is
    described in. Returns the port it drives, along with a handle used to
    declare the subordinates it accesses. The name, if given, names the port
    the manager connects to once lowered.

    Example:
    ```mlir
    %mgr, %access = axi4.dummies.ext_manager "core" %clk, %rst_ni
      addr_width = 32, data_width = 64, outstanding_writes = 4,
      outstanding_reads = 4
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset,
                       OptionalAttr<StrAttr>:$name, I32Attr:$addr_width,
                       I32Attr:$data_width, I32Attr:$outstanding_writes,
                       I32Attr:$outstanding_reads);
  let results = (outs DummiesPortType:$port,
                      DummiesManagerAccessType:$access);

  let assemblyFormat = [{
    ($name^)? $clock `,` $reset `addr_width` `=` $addr_width `,`
      `data_width` `=` $data_width `,`
      `outstanding_writes` `=` $outstanding_writes `,`
      `outstanding_reads` `=` $outstanding_reads attr-dict
  }];

  let hasVerifier = 1;
}

def DummiesExtSubordinateOp : AXI4DummiesOp<"ext_subordinate"> {
  let summary = "An AXI4 subordinate outside the module describing the network";
  let description = [{
    Models a subordinate whose implementation sits outside the module the
    network is described in. Takes the upstream port it responds to and the
    window it handles requests to, and returns a handle used to declare the
    managers that access it. The name, if given, names the port the subordinate
    connects to once lowered.

    Example:
    ```mlir
    %access = axi4.dummies.ext_subordinate "mem" %clk, %rst_ni, %mgr
      window <base = 0x0, last = 0xfff, burst_specs = <<incr, len = 16>>>
      addr_width = 32, data_width = 64, outstanding_writes = 4,
      outstanding_reads = 4
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset,
                       DummiesPortType:$upstream, OptionalAttr<StrAttr>:$name,
                       WindowAttr:$window,
                       I32Attr:$addr_width, I32Attr:$data_width,
                       I32Attr:$outstanding_writes,
                       I32Attr:$outstanding_reads);
  let results = (outs DummiesSubordinateAccessType:$access);

  let assemblyFormat = [{
    ($name^)? $clock `,` $reset `,` $upstream `window` $window
      `addr_width` `=` $addr_width `,` `data_width` `=` $data_width `,`
      `outstanding_writes` `=` $outstanding_writes `,`
      `outstanding_reads` `=` $outstanding_reads attr-dict
  }];

  let hasVerifier = 1;
}

def DummiesXbarOp : AXI4DummiesOp<"xbar"> {
  let summary = "An AXI4 crossbar interconnect";
  let description = [{
    Routes the requests of a set of upstream managers onto downstream
    subordinates. Each use of its result corresponds to a physical downstream
    port.

    Example:
    ```mlir
    %xbar = axi4.dummies.xbar %clk, %rst_ni mgrs %core, %debug addr_width = 32,
      data_width = 64
    ```
  }];

  let arguments = (ins ClockType:$clock, I1:$reset,
                       Variadic<DummiesPortType>:$upstream,
                       I32Attr:$addr_width, I32Attr:$data_width);
  let results = (outs DummiesPortType:$downstream);

  let assemblyFormat = [{
    $clock `,` $reset `mgrs` $upstream `addr_width` `=` $addr_width `,`
      `data_width` `=` $data_width attr-dict
  }];

  let hasVerifier = 1;
}

def DummiesAccessesOp : AXI4DummiesOp<"accesses"> {
  let summary = "Declares that a manager accesses a subordinate";
  let description = [{
    Declares that a manager can access a subordinate with someset of bursts.

    The bursts are the manager's, so their lengths count beats of the manager's
    data width.

    Example:
    ```mlir
    axi4.dummies.accesses %core_access -> %mem_access with <<incr, len = 16>>
    ```
  }];

  let arguments = (ins DummiesManagerAccessType:$manager,
                       DummiesSubordinateAccessType:$subordinate,
                       BurstSetAttr:$bursts);

  let assemblyFormat =
      "$manager `->` $subordinate `with` $bursts attr-dict";
}

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_TD
