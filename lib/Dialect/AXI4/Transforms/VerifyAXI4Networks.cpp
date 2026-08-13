//===- VerifyAXI4Networks.cpp - Verify AXI4 networks ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Verifies the properties of an AXI4 network that span more than one operation.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace axi4 {
#define GEN_PASS_DEF_VERIFYAXI4NETWORKS
#include "circt/Dialect/AXI4/AXI4Passes.h.inc"
} // namespace axi4
} // namespace circt

using namespace circt;
using namespace axi4;
using namespace mlir;

namespace {
/// The clock and reset an AXI4 op operates in.
struct Domain {
  Value clock, reset;
};

/// The domains an AXI4 op takes its upstream ports in and drives its downstream
/// ports in. Only an `axi4.cdc` differs between the two.
struct Domains {
  Domain upstream, downstream;
};
} // namespace

/// The domains of an AXI4 op, or failure for one this pass does not know.
static FailureOr<Domains> getDomains(Operation *op) {
  return TypeSwitch<Operation *, FailureOr<Domains>>(op)
      .Case<AbstractManagerOp, AbstractSubordinateOp, ChannelStructsToPortOp,
            PortToChannelStructsOp, XbarOp, CutOp, DWConverterOp,
            BurstSplitterOp, DemuxOp, MuxOp>([](auto op) {
        Domain domain{op.getClock(), op.getReset()};
        return Domains{domain, domain};
      })
      .Case<CDCOp>([](CDCOp op) {
        // A crossing changes clock but not reset
        return Domains{{op.getUpstreamClock(), op.getReset()},
                       {op.getDownstreamClock(), op.getReset()}};
      })
      .Default([](Operation *op) -> FailureOr<Domains> {
        op->emitOpError("unsupported AXI4 network op; cannot verify which "
                        "clock and reset domain it is in");
        return failure();
      });
}

/// Report a port value with more than one consumer, or with none at all.
static LogicalResult verifyPortUses(Value port) {
  if (!isa<PortType>(port.getType()))
    return success();
  if (port.use_empty()) {
    mlir::emitWarning(port.getLoc())
        << "AXI4 port has no uses, so takes no part in a network";
    return success();
  }
  if (port.hasNUsesOrMore(2))
    return mlir::emitError(port.getLoc())
           << "AXI4 port must have at most one use; route through an "
              "'axi4.xbar' to fan out to multiple endpoints";
  return success();
}

/// Report two ops connected by a port but operating in different domains.
static void emitDomainCrossing(Operation *op, Operation *other,
                               StringRef domain) {
  auto diag = op->emitOpError()
              << "is in a different " << domain << " domain to the '"
              << other->getName().getStringRef() << "' connected to it";
  diag.attachNote(other->getLoc()) << "connected operation here";
}

/// The longest burst `port` supports, in beats.
static uint32_t longestBurst(PortType port) {
  uint32_t most = 0;
  for (WindowAttr window : port.getWindows().getWindows())
    for (BurstSpecAttr spec : window.getBurstSpecs().getBurstSpecs())
      most = std::max(most, spec.getLen());
  return most;
}

/// Report a `downstream` port that cannot hold as many outstanding transactions
/// as the `writes` and `reads` reaching it from upstream. Flow control makes
/// the manager wait, so this costs throughput rather than correctness.
static void warnBottleneck(Operation *op, const Twine &portDesc,
                           PortType downstream, uint64_t writes, uint64_t reads,
                           const Twine &sourceDesc) {
  if (downstream.getOutstandingWrites() < writes)
    op->emitWarning() << portDesc << " can hold fewer outstanding writes than "
                      << sourceDesc << " can issue ("
                      << downstream.getOutstandingWrites() << " < " << writes
                      << ")";
  if (downstream.getOutstandingReads() < reads)
    op->emitWarning() << portDesc << " can hold fewer outstanding reads than "
                      << sourceDesc << " can issue ("
                      << downstream.getOutstandingReads() << " < " << reads
                      << ")";
}

/// Report the downstream ports of a routing op that cannot hold as many
/// outstanding transactions as the upstream ports reaching them can issue.
static void warnRoutingBottlenecks(Operation *op, ValueRange upstream,
                                   ValueRange downstream) {
  for (auto [i, value] : llvm::enumerate(downstream)) {
    auto downstreamTy = cast<PortType>(value.getType());

    uint64_t writes = 0, reads = 0;
    for (Value value : upstream) {
      auto manager = cast<PortType>(value.getType());
      if (!manager.getWindows().overlaps(downstreamTy.getWindows()))
        continue;
      writes += manager.getOutstandingWrites();
      reads += manager.getOutstandingReads();
    }

    warnBottleneck(op, "downstream port #" + Twine(i), downstreamTy, writes,
                   reads, "the managers reaching it");
  }
}

namespace {
struct VerifyAXI4NetworksPass
    : public circt::axi4::impl::VerifyAXI4NetworksBase<VerifyAXI4NetworksPass> {
  void runOnOperation() override;
};
} // namespace

void VerifyAXI4NetworksPass::runOnOperation() {
  ModuleOp module = getOperation();
  Dialect *axi4Dialect = module->getContext()->getLoadedDialect<AXI4Dialect>();
  bool anyFailed = false;

  // Check uses of all axi4.port values
  module.walk([&](Operation *op) {
    for (Value result : op->getResults())
      if (failed(verifyPortUses(result)))
        anyFailed = true;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (failed(verifyPortUses(arg)))
            anyFailed = true;
  });

  // Ensure connected ops are in the same clock and reset domains
  module.walk([&](Operation *op) {
    if (op->getDialect() != axi4Dialect)
      return;
    FailureOr<Domains> domains = getDomains(op);
    if (failed(domains)) {
      anyFailed = true;
      return;
    }

    for (Value operand : op->getOperands()) {
      if (!isa<PortType>(operand.getType()))
        continue;
      // A port arriving from outside the module carries no comparable clock.
      Operation *upstream = operand.getDefiningOp();
      if (!upstream || upstream->getDialect() != axi4Dialect)
        continue;

      FailureOr<Domains> upstreamDomains = getDomains(upstream);
      if (failed(upstreamDomains)) {
        anyFailed = true;
        continue;
      }
      // The port leaves the op that produced it in that op's downstream domain,
      // and arrives in this one's upstream domain.
      const Domain &consumer = domains->upstream;
      const Domain &producer = upstreamDomains->downstream;
      if (consumer.clock != producer.clock) {
        emitDomainCrossing(op, upstream, "clock");
        anyFailed = true;
      }
      if (consumer.reset != producer.reset) {
        emitDomainCrossing(op, upstream, "reset");
        anyFailed = true;
      }
    }
  });

  // Warn on bottlenecks where a downstream port cannot concurrently handle all
  // the possible concurrent requests reaching it
  module.walk([](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<XbarOp, DemuxOp, MuxOp>([](auto routing) {
          warnRoutingBottlenecks(routing, routing.getUpstream(),
                                 routing.getDownstream());
        })
        .Case<DWConverterOp>([](DWConverterOp converter) {
          auto upstream = cast<PortType>(converter.getUpstream().getType());
          warnBottleneck(converter, "downstream port",
                         cast<PortType>(converter.getDownstream().getType()),
                         upstream.getOutstandingWrites(),
                         upstream.getOutstandingReads(), "the upstream port");
        })
        .Case<BurstSplitterOp>([](BurstSplitterOp splitter) {
          auto upstream = cast<PortType>(splitter.getUpstream().getType());
          // A splitter pushes a burst's beats downstream back to back, waiting
          // for none of their responses, so each burst in flight occupies a
          // downstream slot per beat.
          uint64_t beats = longestBurst(upstream);
          warnBottleneck(splitter, "downstream port",
                         cast<PortType>(splitter.getDownstream().getType()),
                         beats * upstream.getOutstandingWrites(),
                         beats * upstream.getOutstandingReads(),
                         "splitting the upstream port's bursts of up to " +
                             Twine(beats) + " beats");
        });
  });

  if (anyFailed)
    signalPassFailure();
}
