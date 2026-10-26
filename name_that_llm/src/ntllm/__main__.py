"""Command-line entry point for the ntllm package.

impl: FR-001-03
impl: FR-001-08
"""

from __future__ import annotations

import argparse
import sys

from ntllm.audit import audit_from_config
from ntllm.data import AuditDataError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ntllm")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="inspect configured local competition data")
    audit.add_argument("--config", required=True, help="path to the YAML audit configuration")
    args = parser.parse_args(argv)

    if args.command == "audit":
        try:
            report = audit_from_config(args.config)
        except AuditDataError as exc:
            print(f"audit: {exc}", file=sys.stderr)
            return 1

        print(f"audit: {report['audit_status']}")
        readiness = report["training_readiness"]
        print(f"training readiness: {readiness['status']}")
        for blocker in readiness["blockers"]:
            print(f"  - {blocker}")
        if report["audit_status"] == "failed":
            for name, file_result in report["files"].items():
                if "error" in file_result:
                    print(f"  - {name}: {file_result['error']}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
