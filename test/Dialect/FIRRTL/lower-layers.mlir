// RUN: circt-opt -firrtl-lower-layers -split-input-file %s | FileCheck %s

firrtl.circuit "Test" {
  firrtl.module @Test() {}

  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind {}
    }
  }
  firrtl.layer @B bind {}

  firrtl.extmodule @Foo(out o : !firrtl.probe<uint<1>, @A>)

  //===--------------------------------------------------------------------===//
  // Removal of Probe Colors
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @ColoredPorts(out %o: !firrtl.probe<uint<1>>)
  firrtl.module @ColoredPorts(out %o: !firrtl.probe<uint<1>, @A>) {}

  // CHECK-LABEL: @ExtColoredPorts(out o: !firrtl.probe<uint<1>>)
  firrtl.extmodule @ExtColoredPorts(out o: !firrtl.probe<uint<1>, @A>)

  // CHECK-LABEL: @ColoredPortsOnInstances
  firrtl.module @ColoredPortsOnInstances() {
    // CHECK: %foo_o = firrtl.instance foo @ColoredPorts(out o: !firrtl.probe<uint<1>>)
   %foo_o = firrtl.instance foo @ColoredPorts(out o: !firrtl.probe<uint<1>, @A>)
  }

  // CHECK-LABEL: @ColoredThings
  firrtl.module @ColoredThings() {
    // CHECK: %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>>
    %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>, @A>
    // CHECK: %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>>
    %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>, @A>
    // CHECK-NOT: firrtl.cast
    %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
  }

    // CHECK-LABEL: @ColoredThingUnderWhen
  firrtl.module @ColoredThingUnderWhen(in %b : !firrtl.uint<1>) {
    // CHECK: firrtl.when %b : !firrtl.uint<1>
    firrtl.when %b : !firrtl.uint<1> {
      // CHECK: %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>>
      %0 = firrtl.wire : !firrtl.probe<bundle<f: uint<1>>, @A>
      // CHECK: %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>>
      %1 = firrtl.ref.sub %0[0] : !firrtl.probe<bundle<f: uint<1>>, @A>
      // CHECK-NOT: firrtl.cast
      %2 = firrtl.ref.cast %1 : (!firrtl.probe<uint<1>, @A>) -> !firrtl.probe<uint<1>, @A::@B>
    }
  }

  //===--------------------------------------------------------------------===//
  // Removal of Enabled Layers
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @EnabledLayers() {
  firrtl.module @EnabledLayers() attributes {layers = [@A]} {}

  // CHECK-LABEL: @EnabledLayersOnInstance()
  firrtl.module @EnabledLayersOnInstance() attributes {layers = [@A]} {
    // CHECK: firrtl.instance enabledLayers @EnabledLayers()
    firrtl.instance enabledLayers {layers = [@A]} @EnabledLayers()
  }

  //===--------------------------------------------------------------------===//
  // Removal of Layerblocks and Layers
  //===--------------------------------------------------------------------===//

  // CHECK-NOT: firrtl.layer @GoodbyeCruelWorld
  firrtl.layer @GoodbyeCruelWorld bind {}

  // CHECK-LABEL @WithLayerBlock
  firrtl.module @WithLayerBlock() {
    // CHECK-NOT firrtl.layerblock @GoodbyeCruelWorld
    firrtl.layerblock @GoodbyeCruelWorld {
    }
  }

  //===--------------------------------------------------------------------===//
  // Capture
  //===--------------------------------------------------------------------===//

  // CHECK: firrtl.module private @[[A:.+]](in %[[x:.+]]: !firrtl.uint<1>, in %[[y:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.add %[[x]], %[[y]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: }
  // CHECK: firrtl.module @CaptureHardware() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]], %[[q:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.strictconnect %[[q]], %c1_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.strictconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureHardware() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %x = firrtl.node %[[p]] : !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.module @CapturePort(in %in: !firrtl.uint<1>) {
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.strictconnect %[[p]], %in : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CapturePort(in %in: !firrtl.uint<1>){
    firrtl.layerblock @A {
      %x = firrtl.node %in : !firrtl.uint<1>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %w = firrtl.wire : !firrtl.uint<1>
  // CHECK:   firrtl.connect %w, %[[p]] : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.module @CaptureHardwareViaConnect() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.strictconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureHardwareViaConnect() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %w = firrtl.wire : !firrtl.uint<1>
      firrtl.connect %w, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @CaptureProbeSrc() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %w = firrtl.wire : !firrtl.uint<1>
  // CHECK:   %0 = firrtl.ref.send %w : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.strictconnect %[[p]], %1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @CaptureProbeSrc() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %w = firrtl.wire : !firrtl.uint<1>
    %r = firrtl.ref.send %w : !firrtl.uint<1>
    firrtl.layerblock @A {
      firrtl.ref.resolve %r : !firrtl.probe<uint<1>>
    }
  }

  // CHECK: firrtl.module private @[[B:.+]](in %[[p:.+]]: !firrtl.uint<1>, in %[[q:.+]]: !firrtl.uint<1>)
  // CHECK:   %0 = firrtl.add %[[p]], %[[q]] : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: }
  // CHECK: firrtl.module private @[[A:.+]](out %[[p:.+]]: !firrtl.probe<uint<1>>, out %[[q:.+]]: !firrtl.probe<uint<1>>) attributes {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %0 = firrtl.ref.send %c0_ui1 : !firrtl.uint<1>
  // CHECK:   firrtl.ref.define %[[q]], %0 : !firrtl.probe<uint<1>>
  // CHECK:   %c0_ui1_1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.send %c0_ui1_1 : !firrtl.uint<1>
  // CHECK:   firrtl.ref.define %[[p]], %1 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @NestedCaptureHardware() {
  // CHECK:   %[[b1:.+]], %[[b2:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A_B.sv", excludeFromFileList>} @[[B]]
  // CHECK:   %[[a1:.+]], %[[a2:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %0 = firrtl.ref.resolve %[[a2]] : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.strictconnect %[[b1]], %0 : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %[[a1]] : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.strictconnect %[[b2]], %1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @NestedCaptureHardware() {
    firrtl.layerblock @A {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      %c1_ui1 = firrtl.constant 0 : !firrtl.uint<1>
      firrtl.layerblock @A::@B {
        %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
      }
    }
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
  // CHECK:   firrtl.when %[[p]] : !firrtl.uint<1> {
  // CHECK:     %0 = firrtl.add %[[p]], %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK:   }
  // CHECK: }
  // CHECK: firrtl.module @WhenUnderLayer() {
  // CHECK:   %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.strictconnect %[[p]], %c0_ui1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @WhenUnderLayer() {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.layerblock @A {
      %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
      firrtl.when %c0_ui1 : !firrtl.uint<1> {
        %0 = firrtl.add %c0_ui1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Connecting/Defining Refs
  //===--------------------------------------------------------------------===//

  // Src and Dst Outside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK: }
  // CHECK: firrtl.module @SrcDstOutside() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A:.+]]()
  // CHECK: }
  firrtl.module @SrcDstOutside() {
    %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>, @A>
    }
  }

  // Src Outside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]](in %_: !firrtl.uint<1>) {
  // CHECK:   %0 = firrtl.ref.send %_ : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @SrcOutside() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.strictconnect %[[p]], %1 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @SrcOutside() {
    %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>, @A>
    }
  }

  // Dst Outside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]](out %[[p:.+]]: !firrtl.probe<uint<1>>) {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %[[p]], %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @DestOutside() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   firrtl.ref.define %0, %[[p]] : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @DestOutside() {
    %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %0, %1 : !firrtl.probe<uint<1>, @A>
    }
  }

  // Src and Dst Inside Layerblock.
  //
  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK:   %0 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %1 = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @SrcDstInside() {
  // CHECK:   firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]()
  // CHECK: }
  firrtl.module @SrcDstInside() {
    firrtl.layerblock @A {
      %0 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      %1 = firrtl.wire : !firrtl.probe<uint<1>, @A>
      firrtl.ref.define %1, %0 : !firrtl.probe<uint<1>, @A>
    }
  }

  //===--------------------------------------------------------------------===//
  // Resolving Colored Probes
  //===--------------------------------------------------------------------===//

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @ResolveColoredRefUnderLayerBlock() {
  // CHECK:   %w = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.strictconnect %[[p]], %0 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @ResolveColoredRefUnderLayerBlock() {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    firrtl.layerblock @A {
      %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module @ResolveColoredRefUnderEnabledLayer() {
  // CHECK:   %w = firrtl.wire : !firrtl.probe<uint<1>>
  // CHECK:   %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @ResolveColoredRefUnderEnabledLayer() attributes {layers=[@A]} {
    %w = firrtl.wire : !firrtl.probe<uint<1>, @A>
    %0 = firrtl.ref.resolve %w : !firrtl.probe<uint<1>, @A>
  }

  // CHECK: firrtl.module private @[[A:.+]](in %[[p:.+]]: !firrtl.uint<1>) {
  // CHECK:   %0 = firrtl.ref.send %[[p]] : !firrtl.uint<1>
  // CHECK:   %1 = firrtl.ref.resolve %0 : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @ResolveColoredRefPortUnderLayerBlock1() {
  // CHECK:   %foo_o = firrtl.instance foo @Foo(out o: !firrtl.probe<uint<1>>)
  // CHECK:   %[[p:.+]] = firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]
  // CHECK:   %0 = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>>
  // CHECK:   firrtl.strictconnect %[[p]], %0 : !firrtl.uint<1>
  // CHECK: }
  firrtl.module @ResolveColoredRefPortUnderLayerBlock1() {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    firrtl.layerblock @A {
      %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module private @[[A:.+]]() {
  // CHECK:   %foo_o = firrtl.instance foo @Foo(out o: !firrtl.probe<uint<1>>)
  // CHECK:   %0 = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>>
  // CHECK: }
  // CHECK: firrtl.module @ResolveColoredRefPortUnderLayerBlock2() {
  // CHECK:   firrtl.instance {{.+}} {lowerToBind, output_file = #hw.output_file<"groups_Test_A.sv", excludeFromFileList>} @[[A]]()
  // CHECK: }
  firrtl.module @ResolveColoredRefPortUnderLayerBlock2() {
    firrtl.layerblock @A {
      %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
      %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
    }
  }

  // CHECK: firrtl.module @ResolveColoredRefPortUnderEnabledLayer() {
  // CHECK:   %foo_o = firrtl.instance foo @Foo(out o: !firrtl.probe<uint<1>>)
  // CHECK:   %0 = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>>
  // CHECK: }
  firrtl.module @ResolveColoredRefPortUnderEnabledLayer() attributes {layers=[@A]} {
    %foo_o = firrtl.instance foo @Foo(out o : !firrtl.probe<uint<1>, @A>)
    %x = firrtl.ref.resolve %foo_o : !firrtl.probe<uint<1>, @A>
  }
}

// -----

firrtl.circuit "Simple" {
  firrtl.layer @A bind {
    firrtl.layer @B bind {
      firrtl.layer @C bind {}
    }
  }

  firrtl.module @Simple() {
    %a = firrtl.wire : !firrtl.uint<1>
    %b = firrtl.wire : !firrtl.uint<2>
    firrtl.layerblock @A {
      %aa = firrtl.node %a: !firrtl.uint<1>
      %c = firrtl.wire : !firrtl.uint<3>
      firrtl.layerblock @A::@B {
        %bb = firrtl.node %b: !firrtl.uint<2>
        %cc = firrtl.node %c: !firrtl.uint<3>
        firrtl.layerblock @A::@B::@C {
          %ccc = firrtl.node %cc: !firrtl.uint<3>
        }
      }
    }
  }
}

// CHECK-LABEL: firrtl.circuit "Simple"
//
// CHECK:      sv.verbatim "`include \22groups_Simple_A.sv\22\0A
// CHECK-SAME:   `include \22groups_Simple_A_B.sv\22\0A
// CHECK-SAME:   `ifndef groups_Simple_A_B_C\0A
// CHECK-SAME:   define groups_Simple_A_B_C"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A_B_C.sv", excludeFromFileList>
// CHECK:      sv.verbatim "`include \22groups_Simple_A.sv\22\0A
// CHECK-SAME:   `ifndef groups_Simple_A_B\0A
// CHECK-SAME:   define groups_Simple_A_B"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A_B.sv", excludeFromFileList>
// CHECK:      sv.verbatim "`ifndef groups_Simple_A\0A
// CHECK-SAME:   define groups_Simple_A"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A.sv", excludeFromFileList>
//
// CHECK:      firrtl.module private @Simple_A_B_C(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[cc_port:[_a-zA-Z0-9]+]]: !firrtl.uint<3>
// CHECK-NEXT:   %ccc = firrtl.node %[[cc_port]]
// CHECK-NEXT: }
//
// CHECK:      firrtl.module private @Simple_A_B(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[b_port:[_a-zA-Z0-9]+]]: !firrtl.uint<2>
// CHECK-SAME:   in %[[c_port:[_a-zA-Z0-9]+]]: !firrtl.uint<3>
// CHECK-SAME:   out %[[cc_port:[_a-zA-Z0-9]+]]: !firrtl.probe<uint<3>>
// CHECK-NEXT:   %bb = firrtl.node %[[b_port]]
// CHECK-NEXT:   %cc = firrtl.node %[[c_port]]
// CHECK-NEXT:   %0 = firrtl.ref.send %cc
// CHECK-NEXT:   firrtl.ref.define %[[cc_port]], %0
// CHECK-NEXT: }
//
// CHECK:      firrtl.module private @Simple_A(
// CHECK-NOT:  firrtl.module
// CHECK-SAME:   in %[[a_port:[_a-zA-Z0-9]+]]: !firrtl.uint<1>
// CHECK-SAME:   out %[[c_port:[_a-zA-Z0-9]+]]: !firrtl.probe<uint<3>>
// CHECK-NEXT:   %aa = firrtl.node %[[a_port]]
// CHECK:        %[[c_ref:[_a-zA-Z0-9]+]] = firrtl.ref.send %c
// CHECK-NEXT:   firrtl.ref.define %[[c_port]], %[[c_ref]]
// CHECK-NEXT: }
//
// CHECK:      firrtl.module @Simple() {
// CHECK-NOT:  firrtl.module
// CHECK-NOT:    firrtl.layerblock
// CHECK:        %[[A_B_C_cc:[_a-zA-Z0-9]+]] = firrtl.instance simple_A_B_C {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"groups_Simple_A_B_C.sv"
// CHECK-SAME:     excludeFromFileList
// CHECK-SAME:     @Simple_A_B_C(
// CHECK-NEXT:   %[[A_B_b:[_a-zA-Z0-9]+]], %[[A_B_c:[_a-zA-Z0-9]+]], %[[A_B_cc:[_a-zA-Z0-9]+]] = firrtl.instance simple_A_B {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"groups_Simple_A_B.sv", excludeFromFileList>
// CHECK-SAME:     @Simple_A_B(
// CHECK-NEXT:   %[[A_B_cc_resolve:[_a-zA-Z0-9]+]] = firrtl.ref.resolve %[[A_B_cc]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_B_C_cc]], %[[A_B_cc_resolve]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_B_b]], %b
// CHECK-NEXT:   %[[A_a:[_a-zA-Z0-9]+]], %[[A_c:[_a-zA-Z0-9]+]] = firrtl.instance simple_A {
// CHECK-SAME:     lowerToBind
// CHECK-SAME:     output_file = #hw.output_file<"groups_Simple_A.sv", excludeFromFileList>
// CHECK-SAME:     @Simple_A(
// CHECK-NEXT:   %[[A_c_resolve:[_a-zA-Z0-9]+]] = firrtl.ref.resolve %[[A_c]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_B_c]], %[[A_c_resolve]]
// CHECK-NEXT:   firrtl.strictconnect %[[A_a]], %a
// CHECK:      }
//
// CHECK-DAG:  sv.verbatim "`endif // groups_Simple_A"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A.sv", excludeFromFileList>
// CHECK-DAG:  sv.verbatim "`endif // groups_Simple_A_B"
// CHECK-SAME:   output_file = #hw.output_file<"groups_Simple_A_B.sv", excludeFromFileList>

// -----

firrtl.circuit "ModuleNameConflict" {
  firrtl.layer @A bind {}
  firrtl.module private @ModuleNameConflict_A() {}
  firrtl.module @ModuleNameConflict() {
    %a = firrtl.wire : !firrtl.uint<1>
    firrtl.instance foo @ModuleNameConflict_A()
    firrtl.layerblock @A {
      %b = firrtl.node %a : !firrtl.uint<1>
    }
  }
}

// CHECK-LABEL: firrtl.circuit "ModuleNameConflict"
//
// CHECK:       firrtl.module private @[[groupModule:[_a-zA-Z0-9]+]](in
//
// CHECK:       firrtl.module @ModuleNameConflict()
// CHECK-NOT:   firrtl.module
// CHECK:         firrtl.instance foo @ModuleNameConflict_A()
// CHECK-NEXT:    firrtl.instance {{[_a-zA-Z0-9]+}} {lowerToBind,
// CHECK-SAME:      @[[groupModule]](