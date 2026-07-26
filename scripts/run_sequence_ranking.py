from __future__ import annotations

import argparse
import json
from pathlib import Path

from streamrank.focused import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the focused KuaiRand sequence-ranking track")
    parser.add_argument("--config", default="configs/sequence_ranking_smoke.json")
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument("--output-dir", help="override the config output directory")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir:
        config["output_dir"] = args.output_dir
    report = run_experiment(config)
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "seed": config.get("seed"),
                "output_dir": config["output_dir"],
                "rows": report["dataset"]["rows"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
