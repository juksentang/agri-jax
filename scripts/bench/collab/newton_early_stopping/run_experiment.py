"""Run the packaged CPU experiment. Install requirements-cpu.txt first."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROFILE = ROOT / "output/profiling"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Check existing results and failure guards")
    mode.add_argument("--smoke", action="store_true", help="Run the 7-day adaptive case with 3 timing repetitions")
    args = parser.parse_args()
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    if args.smoke:
        commands = [["experiment_adaptive_richards.py", "--variant", "adaptive", "--reps", "3", "--tag", "_smoke"]]
    else:
        commands = []
        if not args.check_only:
            for variant, tag in [("fixed", "_final"), ("adaptive", "_final"),
                                 ("fixed", "_repeat"), ("adaptive", "_repeat"), ("unrolled", "_final")]:
                commands.append(["experiment_adaptive_richards.py", "--variant", variant,
                                 "--reps", "51", "--tag", tag])
            commands.append(["experiment_adaptive_richards.py", "--variant", "adaptive", "--audit", "--tag", "_final"])
        commands += [["check_adaptive_failure.py"], ["compare_adaptive_results.py"]]
    for i, command in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run([sys.executable, str(PROFILE / command[0]), *command[1:]],
                       cwd=ROOT, env=env, check=True)
    print("All requested checks passed.", flush=True)


if __name__ == "__main__":
    main()
