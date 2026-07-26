from __future__ import annotations

import argparse
import json

from streamrank.focused import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the focused KuaiRand sequence-ranking track")
    parser.add_argument("--config", default="configs/sequence_ranking_smoke.json")
    args = parser.parse_args()
    report = run_experiment(args.config)
    print(json.dumps({"winner": report["winner"], "rows": report["dataset"]["rows"]}, indent=2))


if __name__ == "__main__":
    main()
