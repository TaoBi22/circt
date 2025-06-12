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
        # if name := re.search(r"%([\s\S]*) :", input):
            # TODO: HACK!!!!!!!!!
            inputNames.append(f"%arg{len(inputNames)}")
    outputs = search.group(3)
    for output in outputs.split(","):
        if width := re.search(r"i([0-9]+)", output):
            outputWidths.append(int(width.group(1)))
    # collect variable types
    x = ""
    while not "fsm.state" in x:
        x = f.readline()
        if search := re.search(r"%([\s\S]+) = fsm.variable [\s\S]+ : i([0-9]+)", x):
            # varNames.append(search.group(1))
            varWidths.append(search.group(2))

# We can do another naughty hack by working out the RTL names of our variables (since they're externalized so just function args)

# The first few args to the circuit function are inputs, then variables, then the time reg, so we can just work out the names
i = len(inputNames) + 3 # 1 for clock, 1 for reset, 1 for state reg
while i < len(inputNames) + len(varWidths) + 3:
    varNames.append(f"arg{i}")
    i += 1

timeRegName = f"arg{i}"

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
        x = re.search(r"hw.output[\s]?(%[A-Za-z0-9_])?(, %[A-Za-z0-9_])+ :", terminator)
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
# os.system(f"cp {builddir}/bmc.mlir {builddir}/safety-tv.mlir")
textToInsert = open(builddir+"/safety.mlir").readlines()[2:-3]
invariants = []
for line in textToInsert:
    if result := re.search(r"(%[a-zA-Z0-9\-_]+) = smt.declare_fun", line):
        invariants.append(result.group(1))

# Declare a function that maps timestep to input
inputFuncDecls = ""
for i, inputWidth in enumerate(inputWidths):
    inputFuncDecls += f"%input_{i}_func = smt.declare_fun \"input_{i}_func\" : !smt.func<(!smt.bv<32>) !smt.bv<{inputWidth}>>\n"

# Add properties that check equivalence
properties = []
for i, invariant in enumerate(invariants):
    propertyStr = f"%tvclause_{i} = smt.forall" + "{\n"
    propertyStr += "^bb0("
    applicationStr = ""
    signatureStr = "!smt.func<("
    bv2Ints = []
    if 1 in inputWidths:
        bv2Ints.append(f"%myConst0 = smt.bv.constant #smt.bv<0> : !smt.bv<1>\n")
        bv2Ints.append(f"%myConst1 = smt.bv.constant #smt.bv<1> : !smt.bv<1>\n")
    for j, inputWidth in enumerate(inputWidths):
        if inputWidth == 1:
            propertyStr += f"%input_{j}: !smt.bool, "
            applicationStr += f"%input_{j}, "
            signatureStr += f"!smt.bool, "
            bv2Ints.append(f"%input_{j}_int = smt.ite %input_{j}, %myConst1, %myConst0 : !smt.bv<{inputWidth}>\n")
        else:
            propertyStr += f"%input_{j}: !smt.bv<{inputWidth}>, "
            applicationStr += f"%input_{j}_int, "
            signatureStr += f"!smt.int, "
            bv2Ints.append(f"%input_{j}_int = smt.bv2int %input_{j} : !smt.bv<{inputWidth}>\n")
    for j, varWidth in enumerate(varWidths):
        propertyStr += f"%var_{j}: !smt.bv<{varWidth}>, "
        applicationStr += f"%var_{j}_int, "
        signatureStr += f"!smt.int, "
        bv2Ints.append(f"%var_{j}_int = smt.bv2int %var_{j} : !smt.bv<{varWidth}>\n")
    for j, outputWidth in enumerate(outputWidths):
        propertyStr += f"%output_{j}: !smt.bv<{outputWidth}>, "
        applicationStr += f"%output_{j}_int, "
        signatureStr += f"!smt.int, "
        bv2Ints.append(f"%output_{j}_int = smt.bv2int %output_{j} : !smt.bv<{outputWidth}>\n")
    propertyStr += "%rtlTime: !smt.bv<32>):\n"
    applicationStr += "%rtlTime_int"
    signatureStr += "!smt.int) !smt.bool>"
    bv2Ints.append(f"%rtlTime_int = smt.bv2int %rtlTime : !smt.bv<32>\n")
    propertyStr += "\n".join(bv2Ints)
    propertyStr += f"%apply = smt.apply_func {invariant}({applicationStr}) : {signatureStr}\n"
    # Make sure that the times match
    propertyStr += f"%rightTime = smt.eq %rtlTime, %{timeRegName} : !smt.bv<32>\n"
    propertyStr += f"%antecedent = smt.and %apply, %rightTime\n"

    # Our form should be: F(I, V, O, T) and T = timeReg and or(V_n != var_n forall n) => false

    # Check equivalence of variables:
    inputChecks = []
    for j, varName in enumerate(varNames):
        propertyStr += f"%var_{j}_eq = smt.eq %var_{j}, %{varName} : !smt.bv<{varWidths[j]}>\n"
        inputChecks.append(f"%var_{j}_eq")

    # Check equivalence of outputs:
    for j, outputName in enumerate(outputNames):
        propertyStr += f"%output_{j}_eq = smt.eq %output_{j}, %{outputName} : !smt.bv<{outputWidths[j]}>\n"
        inputChecks.append(f"%output_{j}_eq")
    
    if (len(inputChecks) == 1):

        # Find the implication
        propertyStr += f"%impl = smt.implies %antecedent, {inputChecks[0]}\n"
    else:
        # AND all our checks
        propertyStr += f"%consequent = smt.and " + ", ".join(inputChecks) + "\n"

        # Find the implication

        propertyStr += f"%impl = smt.implies %antecedent, %consequent\n"

    propertyStr += f"%negatedProp = smt.not %impl\n"
    # And yield it
    propertyStr += f"smt.yield %negatedProp : !smt.bool\n"

    # TODO handle input equivalence
    propertyStr += "}\n"
    propertyStr += f"smt.assert %tvclause_{i}\n"
    properties.append(propertyStr)

# We also have some assertions to make sure that the inputs are what our input function says they should be
# Since inputs are just symbolic vals, this is fine - we just assert equality with the output of our input function
for i, pair in enumerate(zip(inputNames, inputWidths)):
    inputName, inputWidth = pair
    inputFunc = f"%input_{i}_func"
    propertyStr = f"%desired_input_{i} = smt.apply_func {inputFunc}(%{timeRegName}) : !smt.func<(!smt.bv<32>) !smt.bv<{inputWidth}>>\n"
    propertyStr += f"%input_{i}_eq = smt.eq %desired_input_{i}, {inputNames[i]} : !smt.bv<{inputWidth}>\n"
    propertyStr += f"smt.assert %input_{i}_eq\n"
    properties.append(propertyStr)


for i, inputWidth in enumerate(inputWidths):
    inputFunc = f"%input_{i}_func"
    propertyStr = f"%tvclause_input_{i} = smt.forall" + "{\n"
    propertyStr += "^bb0(%time: !smt.bv<32>):\n"
    propertyStr += f"%input_val = smt.apply_func {inputFunc}(%time) : !smt.func<(!smt.bv<32>) !smt.bv<{inputWidth}>>\n"
    propertyStr += f"%input_eq = smt.eq %input_val, {inputNames[i]} : !smt.bv<{inputWidth}>\n"
    propertyStr += "smt.yield %input_eq : !smt.bool\n"
    propertyStr += "}\n"
    propertyStr += f"smt.assert %tvclause_input_{i}\n"
    properties.append(propertyStr)

# TODO: need to add guards to stay in line with the inputs of the RTL

# We need to insert some extra guards into the FSM file to make sure we only take transitions if they're consistent with our expected inputs
fsmTextWithGuards = []

inCHC = False
for line in textToInsert:
    if "smt.forall" in line:
        inCHC = True
    if inCHC:
        if match := re.search(r"%([A-Za-z0-9_]+) = smt.implies %([A-Za-z0-9_]+), %([A-Za-z0-9_]+)", line):
            # This is the implication that we want to add the guard to
            originalCondition = match.group(1)
            originalAntecedent = match.group(2)
            equivalenceChecks = []
            for i, inputWidth in enumerate(inputWidths):
                # We should already have our desired inputs further up in the SMTLIB but in scope here
                equivalenceChecks.append(f"%equivalence_check_{i}")
                fsmTextWithGuards.append(f"%equivalence_check_{i} = smt.eq %obsarg{i}, %desired_input_{i} : !smt.bv<{inputWidth}>\n")
            fsmTextWithGuards.append(f"%equivalence_check = smt.and %{originalAntecedent}, " + ", ".join(equivalenceChecks) + "\n")
            fsmTextWithGuards.append(f"%{originalCondition} = smt.implies %equivalence_check, %{match.group(3)}\n")
            continue


    fsmTextWithGuards.append(line)


bmcText = open(builddir+"/bmc.mlir").readlines()

outputText = []

inCircuitFunc = False
checkResult = None
for line in bmcText:

    if match := re.search(r"%([A-Za-z0-9_]+) = smt\.check", line):
        checkResult = match.group(1)

    if inCircuitFunc and "return" in line:
        outputText.extend(textToInsert)
        for property in properties:
            outputText.append(property)
        inCircuitFunc = False

    outputText.append(line)
    if "func.func @bmc_circuit" in line:
        outputText.append(inputFuncDecls)
        inCircuitFunc = True

    if "} -> i1" in line and checkResult:
        # Insert printing of satisfiability
        
        outputText.append(f"%ss = llvm.mlir.addressof @satString : !llvm.ptr\n")
        outputText.append(f"%us = llvm.mlir.addressof @unsatString : !llvm.ptr\n")
        outputText.append(f"%string = llvm.select %{checkResult}, %ss, %us : i1, !llvm.ptr\n")
        #outputText.append(f"%printf = llvm.mlir.addressof @printf : !llvm.ptr\n")
        outputText.append(f"llvm.call @printf(%string) vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr) -> ()\n")
    
    # Allocate the strings we use to print results
    if "Assertion can be violated" in line:
          outputText.append("llvm.mlir.global private constant @satString(\"sat\\0A\\00\") {addr_space = 0 : i32}\n")
          outputText.append("llvm.mlir.global private constant @unsatString(\"unsat\\0A\\00\") {addr_space = 0 : i32}\n")
  


with open(f"{builddir}/safety-tv.mlir", "w+") as f:
    f.writelines(outputText)

os.system(f"../build/bin/circt-opt --lower-smt-to-z3-llvm {builddir}/safety-tv.mlir --reconcile-unrealized-casts > {builddir}/exec.mlir")
os.system(f"../llvm/build/bin/mlir-cpu-runner {builddir}/exec.mlir -e fsm10 -shared-libs=/usr/lib/x86_64-linux-gnu/libz3.so -entry-point-result=void")
