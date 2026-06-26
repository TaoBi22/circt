//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static analysis to check AXI4 networks are well-formed.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "circt/Dialect/AXI4/AXI4Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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

/// A half-open address range `[first, second)`.
using Interval = std::pair<uint64_t, uint64_t>;

/// An access window propagated downstream from a manager.
struct TrackedWindow {
  uint64_t base, size;
  ArrayRef<BurstSpecAttr> bursts;
  ManagerOp source;
};

static bool overlaps(uint64_t aBase, uint64_t aSize, uint64_t bBase,
                     uint64_t bSize) {
  return aBase < bBase + bSize && bBase < aBase + aSize;
}

/// Coalesce intervals into sorted, maximal, disjoint ranges over the same point
/// set.
static SmallVector<Interval> coalesce(SmallVector<Interval> ivals) {
  llvm::sort(ivals, [](Interval a, Interval b) { return a.first < b.first; });
  SmallVector<Interval> out;
  for (Interval iv : ivals) {
    if (!out.empty() && iv.first <= out.back().second)
      out.back().second = std::max(out.back().second, iv.second);
    else
      out.push_back(iv);
  }
  return out;
}

/// Whether `[lo, hi)` is fully contained in one range of a coalesced set.
static bool covers(ArrayRef<Interval> coalesced, uint64_t lo, uint64_t hi) {
  return llvm::any_of(coalesced, [&](Interval iv) {
    return iv.first <= lo && hi <= iv.second;
  });
}

/// Whether a subordinate's burst specs (`supported`) cover a burst a manager
/// issues (`issued`): same kind, and -- for incrementing bursts -- the exact
/// same length.
static bool supports(ArrayRef<BurstSpecAttr> supported, BurstSpecAttr issued) {
  for (BurstSpecAttr s : supported) {
    if (s.getKind() != issued.getKind())
      continue;
    if (issued.getKind() == BurstKind::Fixed)
      return true; // no length to compare
    if (s.getLen() == issued.getLen())
      return true;
  }
  return false;
}

/// The `!axi4.port` typed operands of `op`.
static SmallVector<Value> portOperands(Operation *op) {
  SmallVector<Value> ports;
  for (Value v : op->getOperands())
    if (isa<PortType>(v.getType()))
      ports.push_back(v);
  return ports;
}

/// The `!axi4.port` typed results of `op`.
static SmallVector<Value> portResults(Operation *op) {
  SmallVector<Value> ports;
  for (Value v : op->getResults())
    if (isa<PortType>(v.getType()))
      ports.push_back(v);
  return ports;
}

/// A forwarder is any interconnect op: it produces a port but isn't a manager.
static bool isForwarder(Operation *op) {
  return op && !isa<ManagerOp>(op) &&
         llvm::any_of(op->getResults(),
                      [](Value v) { return isa<PortType>(v.getType()); });
}

/// Route `op`'s incoming windows onto its port outputs in `valueWindows`.
/// Returns failure (after emitting) for an unknown op or a routing conflict.
static LogicalResult
propagateWindows(Operation *op,
                 DenseMap<Value, SmallVector<TrackedWindow>> &valueWindows) {
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case<XbarOp>([&](XbarOp xbar) -> LogicalResult {
        // An xbar merges its upstream branches; they must be disjoint, else an
        // address is reachable through more than one path.
        SmallVector<TrackedWindow> merged;
        for (Value up : xbar.getUpstream())
          for (TrackedWindow n : valueWindows.lookup(up)) {
            for (TrackedWindow m : merged)
              if (overlaps(n.base, n.size, m.base, m.size))
                return xbar.emitOpError()
                       << "address range [" << n.base << ", " << n.base + n.size
                       << ") is reachable from both "
                       << n.source.getModuleAttr() << " and "
                       << m.source.getModuleAttr();
            merged.push_back(n);
          }
        valueWindows[xbar.getPort()] = std::move(merged);
        return success();
      })
      .Default([](Operation *op) {
        return op->emitOpError("unsupported AXI4 network op; cannot verify how "
                               "it routes addresses");
      });
}

struct VerifyAXI4NetworksPass
    : public circt::axi4::impl::VerifyAXI4NetworksBase<VerifyAXI4NetworksPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool anyError = false;

    // valueWindows(P) = the windows that can arrive on port P.
    DenseMap<Value, SmallVector<TrackedWindow>> valueWindows;

    // Seed manager reaching sets (managers have no port inputs).
    module.walk([&](ManagerOp mgr) {
      SmallVector<TrackedWindow> rs;
      for (Attribute a : mgr.getAccess()) {
        auto w = cast<WindowAttr>(a);
        rs.push_back({w.getBase(), w.getSize(), w.getBurstSpecs(), mgr});
      }
      valueWindows[mgr.getPort()] = std::move(rs);
    });

    // Process forwarders in topological order (Kahn's algorithm), computing
    // their reaching sets and detecting cycles. Only forwarders take port
    // inputs, so only forwarder->forwarder edges can form a cycle.
    SmallVector<Operation *> forwarders;
    DenseMap<Operation *, unsigned> indegree;
    DenseMap<Value, SmallVector<Operation *>>
        consumers; // port -> forwarder users
    SmallVector<Operation *> worklist;
    module.walk([&](Operation *op) {
      if (!isForwarder(op))
        return;
      forwarders.push_back(op);
      unsigned deg = 0;
      for (Value up : portOperands(op))
        if (isForwarder(up.getDefiningOp())) {
          ++deg;
          consumers[up].push_back(op);
        }
      indegree[op] = deg;
      if (deg == 0)
        worklist.push_back(op);
    });

    SmallVector<Operation *> topoOrder;
    while (!worklist.empty()) {
      Operation *fwd = worklist.pop_back_val();
      topoOrder.push_back(fwd);

      // An op we can't route through makes everything downstream unsound; stop.
      if (failed(propagateWindows(fwd, valueWindows)))
        return signalPassFailure();

      for (Value out : portResults(fwd))
        for (Operation *c : consumers[out])
          if (--indegree[c] == 0)
            worklist.push_back(c);
    }

    if (topoOrder.size() != forwarders.size()) {
      for (Operation *fwd : forwarders)
        if (indegree[fwd] != 0)
          fwd->emitOpError("is part of a cyclic AXI4 network");
      return signalPassFailure(); // address analysis below assumes a DAG
    }

    // downstreamSubs(P) = subordinates reachable from port P through forwarders.
    // Computed in reverse topological order so downstream forwarders resolve
    // first.
    DenseMap<Value, SmallVector<SubordinateOp>> downstreamSubs;
    auto computeDownstream = [&](Value port) {
      SmallPtrSet<Operation *, 4> seen;
      SmallVector<SubordinateOp> subs;
      for (Operation *user : port.getUsers()) {
        if (auto sub = dyn_cast<SubordinateOp>(user)) {
          if (seen.insert(sub).second)
            subs.push_back(sub);
        } else if (isForwarder(user)) {
          for (Value out : portResults(user))
            for (SubordinateOp s : downstreamSubs.lookup(out))
              if (seen.insert(s).second)
                subs.push_back(s);
        }
      }
      downstreamSubs[port] = std::move(subs);
    };
    for (Operation *fwd : llvm::reverse(topoOrder))
      for (Value out : portResults(fwd))
        computeDownstream(out);
    module.walk([&](ManagerOp mgr) { computeDownstream(mgr.getPort()); });

    // Per subordinate: the windows on its upstream port are exactly what reaches
    // it, so check exact address alignment and burst support against them.
    module.walk([&](SubordinateOp sub) {
      SmallVector<TrackedWindow> incoming =
          valueWindows.lookup(sub.getUpstream());
      if (incoming.empty())
        return; // upstream not driven by a manager in this module; can't tell

      SmallVector<Interval> issued;
      for (TrackedWindow r : incoming)
        issued.push_back({r.base, r.base + r.size});
      SmallVector<Interval> issuedSet = coalesce(issued);

      for (Attribute a : sub.getAccess()) {
        auto w = cast<WindowAttr>(a);
        uint64_t lo = w.getBase(), hi = w.getBase() + w.getSize();

        // Exact: every handled address must be issued by some reaching manager.
        if (!covers(issuedSet, lo, hi)) {
          sub.emitOpError() << "handles addresses [" << lo << ", " << hi
                            << ") that no manager issues to";
          anyError = true;
        }

        // Burst: each reaching burst that lands in this window must be
        // supported.
        for (TrackedWindow r : incoming) {
          if (!overlaps(r.base, r.size, w.getBase(), w.getSize()))
            continue;
          for (BurstSpecAttr want : r.bursts)
            if (!supports(w.getBurstSpecs(), want)) {
              sub.emitOpError()
                  << "does not support the '"
                  << stringifyBurstKind(want.getKind()) << "' burst issued by "
                  << r.source.getModuleAttr() << " to addresses [" << r.base
                  << ", " << r.base + r.size << ")";
              anyError = true;
            }
        }
      }
    });

    // Per manager: check coverage and unambiguous decode across its whole
    // downstream cone.
    module.walk([&](ManagerOp mgr) {
      // Collect the cone's subordinate windows.
      struct ConeWin {
        uint64_t base, size;
        SubordinateOp op;
      };
      SmallVector<ConeWin> coneWins;
      for (SubordinateOp s : downstreamSubs.lookup(mgr.getPort()))
        for (Attribute a : s.getAccess()) {
          auto w = cast<WindowAttr>(a);
          coneWins.push_back({w.getBase(), w.getSize(), s});
        }

      SmallVector<Interval> handled;
      for (ConeWin c : coneWins)
        handled.push_back({c.base, c.base + c.size});
      SmallVector<Interval> handledSet = coalesce(handled);

      // Coverage: every issued address must reach some subordinate.
      for (Attribute a : mgr.getAccess()) {
        auto w = cast<WindowAttr>(a);
        uint64_t lo = w.getBase(), hi = w.getBase() + w.getSize();
        if (!covers(handledSet, lo, hi)) {
          mgr.emitOpError() << "issues to addresses [" << lo << ", " << hi
                            << ") that are not handled by any subordinate";
          anyError = true;
        }
      }

      // Unambiguous decode: no two cone subordinates may both claim an address
      // this manager issues to.
      auto issues = [&](uint64_t lo, uint64_t hi) {
        for (Attribute a : mgr.getAccess()) {
          auto w = cast<WindowAttr>(a);
          if (overlaps(lo, hi - lo, w.getBase(), w.getSize()))
            return true;
        }
        return false;
      };
      for (unsigned i = 0; i < coneWins.size(); ++i)
        for (unsigned j = i + 1; j < coneWins.size(); ++j) {
          if (!overlaps(coneWins[i].base, coneWins[i].size, coneWins[j].base,
                        coneWins[j].size))
            continue;
          uint64_t lo = std::max(coneWins[i].base, coneWins[j].base);
          uint64_t hi = std::min(coneWins[i].base + coneWins[i].size,
                                 coneWins[j].base + coneWins[j].size);
          if (issues(lo, hi)) {
            mgr.emitOpError() << "issues to addresses [" << lo << ", " << hi
                              << ") that are handled by multiple subordinates";
            anyError = true;
          }
        }
    });

    if (anyError)
      signalPassFailure();
  }
};

} // namespace
