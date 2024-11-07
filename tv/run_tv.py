#!/usr/bin/env python3

import sys, re, os

FSMTRoot = "../../paper-evals/fsm-mc-benchmarking/fsm-circt/"

fsmFile = sys.argv[1]
if len(sys.argv) < 2:
    print("Usage: ./run_tv.py <FSM mlir file>")

moduleName = ""
inputWidths = []
with open(fsmFile, "r") as f:
    x = ""
    while not "fsm.machine" in x:
        x = f.readline()
    search = re.search(r"fsm.machine @([a-zA-Z0-9_\-]+)\(([\s\S]*)\) ->", x)
    moduleName = search.group(1)
    inputs = search.group(2)
    for input in inputs.split(","):
        width = re.search(r": i([0-9]+)", x).group(1)
        inputWidths.append(int(width))

print(inputWidths)
print(moduleName)
builddir = "build"

if os.path.isdir(builddir):
    print("removing existing build directory")
    os.system(f"rm -rf {builddir}")

os.mkdir(builddir)
# BMC file
os.system(f"../build/bin/circt-opt --convert-fsm-to-core {fsmFile} > {builddir}/rtl.mlir")
# get rid of reset values - they're bodged in in this version of the tool
os.system(f"sed -i -E \"s/reset %.+, %.+ :/:/g\" {builddir}/rtl.mlir")
os.system(f"../build/bin/circt-opt --externalize-registers --lower-to-bmc=\"top-module=fsm10 bound=10\" --convert-hw-to-smt --convert-comb-to-smt --convert-verif-to-smt --canonicalize {builddir}/rtl.mlir > {builddir}/bmc.mlir")

# FSM files
os.system(f"{FSMTRoot}/build/bin/circt-opt --convert-fsm-to-smt-safety {fsmFile} > {builddir}/safety.mlir")
os.system(f"{FSMTRoot}/build/bin/circt-opt --convert-fsm-to-smt-safety {fsmFile} > {builddir}/liveness.mlir")

# modify names in FSM files to avoid name conflicts
os.system(f"sed -i \"s/%/%obs/g\" {builddir}/safety.mlir")
os.system(f"sed -i \"s/%/%obs/g\" {builddir}/liveness.mlir")

# setup safety
os.system(f"cp {builddir}/bmc.mlir {builddir}/safety-tv.mlir")
