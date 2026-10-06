//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the passes for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4PASSES_TD
#define CIRCT_DIALECT_AXI4_AXI4PASSES_TD

include "mlir/Pass/PassBase.td"

def VerifyAXI4Networks : Pass<"verify-axi4-networks", "mlir::ModuleOp"> {
  let summary = "Globally verify AXI4 networks";
  let description = [{
    Verifies non-local properties of an AXI4 network:
    - Ensures every `!axi4.port` value has at most one consumer, and that ports
      do not connect endpoints in different clock or reset domains.
    - Ensures a network occupies a single clock and reset domain.
    - Warns where an endpoint can handle fewer requests per ID than the port
      reaching it can have concurrently outstanding. This isn't a correctness
      condition, but flags bottlenecks that might limit throughput.
  }];

  let dependentDialects = ["circt::axi4::AXI4Dialect"];
}

def OptimizeAXI4Networks : Pass<"optimize-axi4-networks", "mlir::ModuleOp"> {
  let summary = "Optimize AXI4 networks";
  let description = [{
    Rewrites a network in ways that keep the data it carries, but not
    necessarily the cycles or the ordering it carries them with, so they are not
    canonicalizations.

    - Removes the downstream ports of `axi4.xbar` and `axi4.demux` operations
      that no upstream manager can address, along with the ops downstream of
      them. Where such a port feeds something this pass will not erase - an
      interface bridged out into HW logic, a memory interface driving live
      logic, or another routing operation - it warns and leaves the branch
      alone.
    - Fuses each adaptor with the like adaptor driving it, if there is one,
      through the cuts and crossings in between, which take the fused
      adaptor's port. Narrowing ID widths merges the IDs in flight, so a fused
      pair can leave transactions free to complete in an order the original
      ordered.
    - Fuses each `axi4.cdc` with the crossing driving it, if there is one, so
      a connection crosses once into the domain it ends up in, rather than once
      per crossing written.
  }];

  let dependentDialects = ["circt::axi4::AXI4Dialect"];
}

def LowerAXI4DummiesToAXI : Pass<"lower-axi4-dummies-to-axi", "mlir::ModuleOp"> {
  let summary = "Lower the dummies subdialect to the AXI4 dialect";
  let description = [{
    Rewrites a network described with the `axi4.dummies` operations into the
    AXI4 dialect, inferring the parameterisation they leave out: the access
    windows of each manager from the accesses it declares, the ID widths from
    the IDs each endpoint has outstanding, and the placement and sizing of the
    width converters needed where an endpoint and the interconnect it connects
    to disagree.

    The external endpoints become ports of the module the network is described
    in, so that module must not be instantiated.

    Each access is routed along the one path from its manager to the
    subordinate it targets that passes through no crossbar twice. A
    crossbar's downstream port carries the windows of the subordinates the
    accesses routed through it target, or of every subordinate below it if
    none is. The network may loop, so long as every loop passes through an
    `axi4.dummies.id_remap`.

    Every port in the network carries the user width given by `user-width`.
    The PULP atop markers `pulp.atops` and `pulp.atop_filter` on an external
    endpoint are put on the port it becomes.

    A crossbar whose upstream ports do not all reach every downstream port is
    given a `PULP_CONFIG_Connectivity`, connecting each upstream port only to
    the downstream ports its managers access through it, unless it already
    has one.
  }];

  let options = [
    Option<"userWidth", "user-width", "uint32_t", "0",
           "The user width of every port in the network">
  ];

  let dependentDialects = ["circt::axi4::AXI4Dialect", "circt::hw::HWDialect"];
}

#endif // CIRCT_DIALECT_AXI4_AXI4PASSES_TD
