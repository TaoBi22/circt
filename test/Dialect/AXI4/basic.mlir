// RUN: circt-opt %s | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !axi4.clock
%c = unrealized_conversion_cast to !axi4.clock

// CHECK: unrealized_conversion_cast to !axi4.reset
%r = unrealized_conversion_cast to !axi4.reset

// CHECK: unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4, 4, 0, 0, 0>
