"""Run every example in order; stop at the first failure.

    python examples/run_all.py                    # figures -> examples/outputs/ (git-ignored)
    python examples/run_all.py --update-figures   # figures -> examples/figures/ (tracked, shown in the README)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--update-figures", action="store_true",
                    help="write figures to examples/figures/ instead of examples/outputs/")
args = parser.parse_args()

env = dict(os.environ)
if args.update_figures:
    env["LS_EXAMPLES_OUT"] = str(HERE / "figures")

scripts = sorted(HERE.glob("ex[0-9][0-9]_*.py"))
for script in scripts:
    t0 = time.time()
    print(f"=== {script.name}", flush=True)
    proc = subprocess.run([sys.executable, script.name], cwd=HERE, env=env)
    print(f"    {'ok' if proc.returncode == 0 else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
print(f"All {len(scripts)} examples passed.")
