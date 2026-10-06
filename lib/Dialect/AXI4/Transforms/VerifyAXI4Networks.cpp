//===----------------------------------------------------------------------===//
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
            PortToChannelStructsOp, XbarOp, CutOp, DWConverterOp, IWConverterOp,
            IDRemapOp, BurstSplitterOp, BurstUnwrapperOp, DemuxOp, MuxOp,
            ToMemOp>([](auto op) {
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
           << "AXI4 port must have at most one use; route through an xbar to "
              "fan out to multiple endpoints";
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

/// Report where `subject`, an endpoint or the upstream port of an op, can
/// handle fewer requests per ID than the port reaching it can have concurrently
/// outstanding - this is a warning since it will only impact throughput.
static void warnBottleneck(Operation *op, const Twine &subject,
                           PortType reaching, uint64_t writes, uint64_t reads) {
  if (writes < reaching.getConcurrentWritesPerId())
    op->emitWarning() << subject
                      << " can handle fewer writes per ID than the port "
                         "reaching it can have concurrently outstanding ("
                      << writes << " < " << reaching.getConcurrentWritesPerId()
                      << ")";
  if (reads < reaching.getConcurrentReadsPerId())
    op->emitWarning() << subject
                      << " can handle fewer reads per ID than the port "
                         "reaching it can have concurrently outstanding ("
                      << reads << " < " << reaching.getConcurrentReadsPerId()
                      << ")";
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

  // Warn on bottlenecks where an endpoint, or an op's budget, may not be able
  // to keep up with the requests reaching it
  module.walk([](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<AbstractSubordinateOp>([](AbstractSubordinateOp subordinate) {
          warnBottleneck(subordinate, "endpoint",
                         subordinate.getUpstream().getType(),
                         subordinate.getConcurrentWritesPerId(),
                         subordinate.getConcurrentReadsPerId());
        })
        .Case<PortToChannelStructsOp>([](PortToChannelStructsOp bridge) {
          warnBottleneck(bridge, "endpoint", bridge.getPort().getType(),
                         bridge.getConcurrentWritesPerId(),
                         bridge.getConcurrentReadsPerId());
        })
        .Case<XbarOp>([](XbarOp xbar) {
          uint32_t budget = xbar.getUpstreamConcurrentPerId();
          for (auto [i, port] : llvm::enumerate(xbar.getUpstream()))
            warnBottleneck(xbar, "upstream port #" + Twine(i),
                           cast<PortType>(port.getType()), budget, budget);
        })
        .Case<DemuxOp>([](DemuxOp demux) {
          uint32_t budget = demux.getUpstreamConcurrentPerId();
          warnBottleneck(demux, "upstream port", demux.getUpstream().getType(),
                         budget, budget);
        })
        .Case<IDRemapOp>([](IDRemapOp remap) {
          uint32_t budget = remap.getConcurrentPerId();
          warnBottleneck(remap, "upstream port", remap.getUpstream().getType(),
                         budget, budget);
        })
        .Case<IWConverterOp>([](IWConverterOp converter) {
          // Widening tracks nothing
          PortType upstream = converter.getUpstream().getType();
          PortType downstream = converter.getDownstream().getType();
          uint64_t writes = upstream.getConcurrentWritesPerId();
          uint64_t reads = upstream.getConcurrentReadsPerId();
          if (downstream.getWriteIdWidth() < upstream.getWriteIdWidth())
            writes = converter.getConcurrentPerId();
          if (downstream.getReadIdWidth() < upstream.getReadIdWidth())
            reads = converter.getConcurrentPerId();
          warnBottleneck(converter, "upstream port", upstream, writes, reads);
        })
        .Case<BurstSplitterOp, BurstUnwrapperOp>([](auto adaptor) {
          warnBottleneck(
              adaptor, "upstream port", adaptor.getUpstream().getType(),
              adaptor.getConcurrentWrites(), adaptor.getConcurrentReads());
        })
        .Case<DWConverterOp>([](DWConverterOp converter) {
          // A conversion converts one read per ID at a time
          PortType upstream = converter.getUpstream().getType();
          if (upstream.getDataWidth() ==
              converter.getDownstream().getType().getDataWidth())
            return;
          warnBottleneck(converter, "upstream port", upstream,
                         upstream.getConcurrentWritesPerId(), 1);
        });
  });

  if (anyFailed)
    signalPassFailure();
}
