import matplotlib.pyplot as plt
import numpy as np
import os
import json
from math import log

configs = ["small", "medium", "large", "dual-small", "dual-medium", "dual-large"]

modRatios = []

variadicCombs = ["add", "mul", "or", "and", "xor", "concat", "truth_table"]
operandMap = {}
for op in variadicCombs:
    operandMap["comb." + op] = {}

for config in configs:
    plt.clf()
    os.system(f"build/bin/circt-opt --convert-verif-to-smt /anfs/bigdisc/tah56/arc-tests/rocket/build/{config}-master/rocket.mlir > /dev/null")
    freqs = {}
    x = []
    os.system(f"cp ops.txt {config}-ops.txt")
    os.system(f"cp ops.json {config}-ops.json")
    os.system(f"cp dump.txt {config}-dump.txt")
    with open("modSizes.txt") as file:
        for line in file:
            num = int(line.strip())
            x.append(num)
            if num in freqs:
                freqs[num] += 1
            else:
                freqs[num] = 1

    # print(x)
    # plt.hist(x, width = 2)
    # plt.savefig("hist.pdf")

    bodge = []
    for i in range(max(x)):
        j = 0
        for item in x:
            if item <= i:
                j += 1
        bodge += [i] * j
    plt.hist(bodge, width = 1, bins = max(x))
    # print(bodge)
    plt.title(f"Module sizes in the {config} Rocket config")
    plt.xlabel("Threshold")
    plt.ylabel("Number of modules with total op count below threshold")
    plt.tight_layout()
    plt.savefig(f"{config}-cumhist.pdf")
    
    content = {}
    with open("ops.json") as file:
        text = file.read()
        content = json.loads(text)
    # print(content)
    plt.clf()
    
    ops = sorted(content.keys())
    quants = []
    # print(ops)
    totals = {}
    for op in ops:
        total = 0
        for num in content[op]:
            total += content[op][num]
        quants.append(total)
        totals[op] = total
    
    plt.xticks(rotation=-45, ha = "left")
    plt.bar(ops, quants)
    plt.title(f"Operation counts in the {config} Rocket config")
    plt.xlabel("Operation name")
    plt.ylabel("Quantity of operation")
    plt.tight_layout()
    plt.savefig(f"{config}-opcounts.pdf")
    
    modRatios.append(totals["hw.instance"]/totals["hw.module"])
    for op in content:
        if op in operandMap:
            for numstr in content[op]:
                num = int(numstr)
                operandMap[op][num] = content[op][numstr]
                
plt.clf()
plt.xticks(rotation=-45, ha = "left")
plt.bar(configs, modRatios)
plt.title(f"hw.instance to hw.module ratio")
plt.xlabel("Rocket config")
plt.ylabel("Number of hw.instance ops / Number of hw.module ops")
plt.tight_layout()
plt.savefig(f"instance-module-ratio.pdf")

totalOperands = {}
for op in operandMap:
    for num in operandMap[op]:
        if num in totalOperands:
            totalOperands[num] += operandMap[op][num]
        else:
            totalOperands[num] = operandMap[op][num]
            
bodge = []
for num in totalOperands:
    bodge += [num] * totalOperands[num]

plt.clf()
plt.hist(bodge, width = 1, bins = max(bodge))
# print(bodge)
plt.title(f"Number of operands used across Rocket configs for variadic comb ops")
plt.yscale('log')
plt.xlabel("Number of operands")
plt.ylabel("Total frequency across Rocket configs")
plt.tight_layout()
plt.savefig(f"variadic-operand-count.pdf")
