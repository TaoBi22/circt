#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("btor")
parser.add_argument("-t", "--test", required=True)
parser.add_argument("-d", "--directory", required=True)
parser.add_argument("-m", "--mode")
parser.add_argument("-k", "--depth", type=int)
args = parser.parse_args()

# Run circt-bmc. We currently have to capture the output and look for a specific
# string to know if the verification passed or failed. Once the tool properly
# exits on test failure, we can skip this.
cmd = ["btormc", args.test]
cmd += ["--bound-max", str(args.depth or 20)]

sys.stderr.write("# " + shlex.join(str(c) for c in cmd) + "\n")
sys.stderr.flush()
print(cmd)
with open(Path(args.directory) / "output.log", "w") as output:
  result = subprocess.call(cmd, stdout=output, stderr=output)
with open(Path(args.directory) / "output.log", "r") as output:
  output_str = output.read()
sys.stderr.write(output_str)
if result != 0:
  sys.exit(result)
sys.exit(0 if len(output_str) == 0 else 1)
