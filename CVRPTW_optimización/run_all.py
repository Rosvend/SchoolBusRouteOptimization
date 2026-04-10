"""
run_all.py  —  Run the full pipeline in one command.

Usage:
    python run_all.py           # runs all 3 stages
    python run_all.py --stage 2 # runs only stage 2 (optimize_routes)
    python run_all.py --stage 3 # runs only stage 3 (visualize_results)
"""

import sys
import time
import argparse


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


def run_stage(name: str, module_path: str) -> None:
    print(f"\n{'#'*62}")
    print(f"#  {name}")
    print(f"{'#'*62}\n")
    t0 = time.time()
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("_stage", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
    print(f"\n  ✓  {name} finished in {hms(time.time() - t0)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="Run only one stage (1, 2, or 3). Default: all.")
    args = parser.parse_args()

    total_t0 = time.time()

    stages = {
        1: ("Stage 1 — Data Generation",    "generate_data.py"),
        2: ("Stage 2 — Route Optimisation", "optimize_routes.py"),
        3: ("Stage 3 — Visualisation",      "visualize_results.py"),
    }

    to_run = [args.stage] if args.stage in stages else [1, 2, 3]

    for s in to_run:
        run_stage(*stages[s])

    print(f"\n{'='*62}")
    print(f"  Pipeline complete in {hms(time.time() - total_t0)}")
    print(f"  Maps saved to: outputs/maps/")
    print(f"  Results:       outputs/route_results.json")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
