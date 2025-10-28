import time, os

def timed_system(cmd: str):
    start = time.perf_counter()
    os.system(cmd)
    end = time.perf_counter()
    return end - start

with open("benchmark_results_50.txt", "w") as resultsFile:
    resultsFile.write("State Count, Time (s)\n")
    for sCount in range(10, 60, 10):
        t = timed_system(f"python3 run_tv.py benchmarks/fsm_{sCount}states_0loops.mlir 50")
        resultsFile.write(f"{sCount}, {t}\n")

