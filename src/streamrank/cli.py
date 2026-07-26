from __future__ import annotations

import argparse
import json

from streamrank.data.audit import audit_interactions, label_cross_table
from streamrank.data.download import download_kuairand_pure
from streamrank.data.loader import load_interactions
from streamrank.data.prepare import prepare_kuairand_pure
from streamrank.serving.build import build_serving_deployment
from streamrank.serving.manifest import DeploymentManifest


def command_audit(args: argparse.Namespace) -> int:
    events = list(load_interactions(args.csv))
    report = audit_interactions(events).to_dict()
    report["long_view_x_is_click"] = label_cross_table(events, "long_view", "is_click")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_manifest(args: argparse.Namespace) -> int:
    manifest = DeploymentManifest.load(args.path)
    print(json.dumps(manifest.__dict__, indent=2))
    return 0


def command_prepare_kuairand(args: argparse.Namespace) -> int:
    report = prepare_kuairand_pure(
        args.raw_dir,
        args.output_dir,
        users=args.users,
        seed=args.seed,
        archive_md5=args.archive_md5,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def command_build_deployment(args: argparse.Namespace) -> int:
    manifest_path = build_serving_deployment(
        args.root,
        catalog_path=args.catalog,
        policy_path=args.policy,
        manifest_name=args.name,
    )
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return 0


def command_download_kuairand(args: argparse.Namespace) -> int:
    report = download_kuairand_pure(args.destination)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="streamrank")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="audit KuaiRand label semantics and data quality")
    audit.add_argument("csv")
    audit.set_defaults(func=command_audit)
    manifest = subparsers.add_parser("manifest", help="validate a deployment manifest")
    manifest.add_argument("path")
    manifest.set_defaults(func=command_manifest)
    prepare = subparsers.add_parser(
        "prepare-kuairand",
        help="build a metadata-enriched deterministic KuaiRand-Pure cohort",
    )
    prepare.add_argument("raw_dir")
    prepare.add_argument("output_dir")
    prepare.add_argument("--users", type=int, default=500, help="0 selects all users")
    prepare.add_argument("--seed", type=int, default=2026)
    prepare.add_argument("--archive-md5")
    prepare.set_defaults(func=command_prepare_kuairand)
    deployment = subparsers.add_parser(
        "build-deployment",
        help="bind the offline-selected ranker, catalog and explicit policy into a manifest",
    )
    deployment.add_argument("--root", default=".")
    deployment.add_argument("--catalog", required=True)
    deployment.add_argument("--policy", default="configs/serving_policy.json")
    deployment.add_argument("--name", default="kuairand-pure-sample")
    deployment.set_defaults(func=command_build_deployment)
    download = subparsers.add_parser(
        "download-kuairand",
        help="download, verify and safely extract official KuaiRand-Pure",
    )
    download.add_argument("destination", nargs="?", default="data/raw")
    download.set_defaults(func=command_download_kuairand)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
