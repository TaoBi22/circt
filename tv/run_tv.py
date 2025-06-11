#!/usr/bin/env python3

import sys, re, os

FSMTRoot = "../../paper-evals/fsm-mc-benchmarking/fsm-circt/"

if len(sys.argv) < 2:
    print("Usage: ./run_tv.py <FSM mlir file>")
    sys.exit(-1)

fsmFile = sys.argv[1]

moduleName = ""
# Fetch the widths of our various values for signatures etc., and their SSA names for comparison
inputWidths = []
inputNames = []
outputWidths = []
outputNames = []
varWidths = []
varNames = []
timeWidth = 32
with open(fsmFile, "r") as f:
    x = ""
    while not "fsm.machine" in x:
        x = f.readline()
    search = re.search(r"fsm.machine @([a-zA-Z0-9_\-]+)\(([\s\S]*)\) -> \(([\s\S]*)\)", x)
    moduleName = search.group(1)
    inputs = search.group(2)
    for input in inputs.split(","):
        if width := re.search(r": i([0-9]+)", input):
            inputWidths.append(int(width.group(1)))
        if name := re.search(r"%([\s\S]*) :", input):
            inputNames.append(int(width.group(1)))
    outputs = search.group(3)
    for output in outputs.split(","):
        if width := re.search(r"i([0-9]+)", output):
            outputWidths.append(int(width.group(1)))
    # collect variable types
    x = ""
    while not "fsm.state" in x:
        x = f.readline()
        if search := re.search(r"%([\s\S]+) = fsm.variable [\s\S]+ : i([0-9]+)", x):
            varNames.append(search.group(1))
            varWidths.append(search.group(2))

assert len(inputWidths) == len(inputNames), "Input widths and names do not match"
assert len(outputWidths) == len(outputNames), "Output widths and names do not match"
assert len(varWidths) == len(varNames), "Variable widths and names do not match"

print(inputWidths)
print(moduleName)
builddir = "build"

if os.path.isdir(builddir):
    print("removing existing build directory")
    os.system(f"rm -rf {builddir}")

os.mkdir(builddir)
# BMC file
os.system(f"../build/bin/circt-opt --convert-fsm-to-core {fsmFile} > {builddir}/untimed_rtl.mlir")
# Insert timer circuit
with open(f"{builddir}/untimed_rtl.mlir") as file:
    text = file.readlines()[:-3]
    terminator = text[-1]
    del[text[-1]]
    # Timer circuit has to be last reg in module so we know where it'll appear in circuit signature
    text += [    "%mySpecialConstant = hw.constant 1 : i32\n",
    "%time_reg = seq.compreg sym @time_reg %added, %clk : i32\n",
    "%added = comb.add %time_reg, %mySpecialConstant : i32\n"]
    text.append(terminator)
    text += ["}"]*2
    with open(f"{builddir}/rtl.mlir", "w+") as newFile:
        newFile.writelines(text)

    # Since we have this text knocking around anyway, we might as well grab the output SSA names
    # First check we actually have outputs
    if terminator.strip() != "hw.output":
        x = re.search("hw.output[\s]?(%[A-Za-z0-9_])?(, %[A-Za-z0-9_])+ :", terminator)
        outputNames.append(x.group(0))
        if x.group(1):
            outputNames = [x.group(1).split(",")]

print("Output names:", outputNames)

os.system(f"../build/bin/circt-opt --externalize-registers --lower-to-bmc=\"top-module=fsm10 bound=10\" --convert-hw-to-smt --convert-comb-to-smt --convert-verif-to-smt --canonicalize {builddir}/rtl.mlir > {builddir}/bmc.mlir")

# FSM files
os.system(f"{FSMTRoot}/build/bin/circt-opt --convert-fsm-to-smt-safety {fsmFile} > {builddir}/safety.mlir")
os.system(f"{FSMTRoot}/build/bin/circt-opt --convert-fsm-to-smt-safety {fsmFile} > {builddir}/liveness.mlir")

# modify names in FSM files to avoid name conflicts
os.system(f"sed -i \"s/%/%obs/g\" {builddir}/safety.mlir")
os.system(f"sed -i \"s/%/%obs/g\" {builddir}/liveness.mlir")

# setup safety
os.system(f"cp {builddir}/bmc.mlir {builddir}/safety-tv.mlir")
textToInsert = open(builddir+"/safety.mlir").readlines()[2:-3]
invariants = []
for line in textToInsert:
    if result := re.search(r"(%[a-zA-Z0-9\-_]+) = smt.declare_fun", line):
        invariants.append(result.group(1))

# Add properties that check equivalence
properties = []
for i, invariant in enumerate(invariants):
    propertyStr = f"%tvclause_{i} = smt.forall" + "{\n"
    propertyStr += "^bb0("
    applicationStr = ""
    signatureStr = "!smt.func<("
    for j, inputWidth in enumerate(inputWidths):
        propertyStr += f"%input_{j}: !smt.bv<{inputWidth}>, "
        applicationStr += f"%input_{j}, "
        signatureStr += f"!smt.bv<{inputWidth}>, "
    for j, varWidth in enumerate(varWidths):
        propertyStr += f"%var_{j}: !smt.bv<{varWidth}>, "
        applicationStr += f"%var_{j}, "
        signatureStr += f"!smt.bv<{varWidth}>, "
    for j, outputWidth in enumerate(outputWidths):
        propertyStr += f"%output_{j}: !smt.bv<{outputWidth}>, "
        applicationStr += f"%output_{j}, "
        signatureStr += f"!smt.bv<{outputWidth}>, "
    propertyStr += "%rtlTime: !smt.bv<32>):\n"
    applicationStr += "%rtlTime"
    signatureStr += "!smt.bv<32>) !smt.bool>"
    propertyStr += f"%apply = smt.apply_func {invariant}({applicationStr}) : {signatureStr}\n"
    # Make sure that the times match
    propertyStr += f"%rightTime = smt.eq %rtlTime, %time_reg : !smt.bv<32>\n"
    propertyStr += f"%antecedent = smt.and %apply, %rightTime : !smt.bool\n"

    # Check equivalence of variables:
    inputChecks = []
    for j, varName in enumerate(varNames):
        propertyStr += f"%var_{j}_eq = smt.eq %var_{j}, %{varName} : !smt.bv<{varWidths[j]}>\n"
        inputChecks.append(f"%var_{j}_eq")

    # Check equivalence of outputs:
    for j, outputName in enumerate(outputNames):
        propertyStr += f"%output_{j}_eq = smt.eq %output_{j}, %{outputName} : !smt.bv<{outputWidths[j]}>\n"
        inputChecks.append(f"%output_{j}_eq")

    # AND all our checks
    propertyStr += f"%consequent = smt.and " + ", ".join(inputChecks) + " : !smt.bool\n"

    # Find the implication

    propertyStr += f"%impl = smt.implies %antecedent, %consequent : !smt.bool\n"

    # And yield it
    propertyStr += f"smt.yield %impl : !smt.bool\n"

    # TODO handle input equivalence
    propertyStr += "}\n"
    propertyStr += f"smt.assert %tvclause_{i}"
    print(propertyStr)
