import sys
from pathlib import Path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
import benchmark, explain, gate, leakage, write_report

STAGES = dict(leakage=leakage.main, benchmark=benchmark.main, gate=gate.main, explain=explain.main, report=write_report.main)

if __name__ == "__main__":
    for name in (sys.argv[1:] or STAGES):
        print(f"== {name}", flush=True)
        STAGES[name]()