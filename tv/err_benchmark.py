import time, os
import sys

def timed_system(cmd: str):
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    return end - start

reps = 3

with open("err_benchmark_results_10.txt", "w") as resultsFile:
    resultsFile.write("states,time\n")
    for sCount in range(10, 60, 10):
        for i in range(reps):
            print(f"rep {i}, scount {sCount}")
            t = timed_system(f"timeout 150m python3 run_tv.py err_benchmarks/fsm_{sCount}states_0loops.mlir 10")
            if t > 8900:
                break
            resultsFile.write(f"{sCount}, {t}\n")
            resultsFile.flush()
            os.fsync(resultsFile)

with open("err_benchmark_results_50.txt", "w") as resultsFile:
    resultsFile.write("states,time\n")
    for sCount in range(10, 60, 10):
        for i in range(reps):
            print(f"rep {i}, scount {sCount}")
            t = timed_system(f"timeout 150m python3 run_tv.py err_benchmarks/fsm_{sCount}states_0loops.mlir 50")
            if t > 8900:
                break
            resultsFile.write(f"{sCount}, {t}\n")
            resultsFile.flush()
            os.fsync(resultsFile)
