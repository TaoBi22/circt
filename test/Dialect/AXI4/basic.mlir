// RUN: circt-opt %s | circt-opt | FileCheck %s

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !axi4.port<32, 64, 4>
%p = unrealized_conversion_cast to !axi4.port<32, 64, 4>
