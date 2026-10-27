"""Command-line entry point for the ntllm package.

impl: FR-001-03
impl: FR-001-08
impl: FR-002-06
impl: FR-002-08
impl: FR-003-06, FR-003-08
impl: FR-004-01, FR-004-09
impl: FR-004-04, FR-004-06
impl: FR-006-01, FR-006-09
"""

from __future__ import annotations

import argparse
import sys

from ntllm.audit import audit_from_config
from ntllm.analysis import AnalysisError, analyze_oof_run, tune_from_config
from ntllm.data import AuditDataError
from ntllm.inference import (
    InferenceError,
    default_submission_output,
    fit_final_from_config,
    predict_from_artifact,
    validate_submission_from_config,
)
from ntllm.splits import SplitError, split_from_config
from ntllm.training import TrainingError, train_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ntllm")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="inspect configured local competition data")
    audit.add_argument("--config", required=True, help="path to the YAML audit configuration")
    split = commands.add_parser("split", help="create prompt-grouped validation folds")
    split.add_argument("--config", required=True, help="path to the YAML split configuration")
    train = commands.add_parser("train", help="run grouped out-of-fold baseline benchmarks")
    train.add_argument("--config", required=True, help="path to the YAML training configuration")
    fit_final = commands.add_parser("fit-final", help="fit the selected model on all audited train rows")
    fit_final.add_argument("--config", required=True, help="path to the YAML final-fit configuration")
    predict = commands.add_parser("predict", help="predict labels offline from a saved model bundle")
    predict.add_argument("--model", required=True, help="path to the final model bundle")
    predict.add_argument("--input", required=True, help="test CSV path")
    predict.add_argument("--output", help="submission Parquet output path; defaults to the config value")
    predict.add_argument("--config", default="configs/baseline.yaml", help="YAML config for the default output path")
    validate = commands.add_parser("validate-submission", help="validate a Kaggle submission file")
    validate.add_argument("--input", required=True, help="submission Parquet path")
    validate.add_argument("--test", required=True, help="test CSV path")
    validate.add_argument("--config", default="configs/baseline.yaml", help="YAML config with official labels")
    evaluate = commands.add_parser("evaluate", help="summarize out-of-fold errors for a run")
    evaluate.add_argument("--run-id", required=True, help="grouped OOF run ID")
    evaluate.add_argument("--candidate", help="candidate name (defaults to configured inference model)")
    evaluate.add_argument("--config", default="configs/baseline.yaml", help="baseline YAML path")
    tune = commands.add_parser("tune", help="run a bounded candidate grid on the saved folds")
    tune.add_argument("--config", default="configs/baseline.yaml", help="baseline YAML path")
    tune.add_argument("--experiment", required=True, help="versioned experiment YAML path")
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

    if args.command == "split":
        try:
            report = split_from_config(args.config)
        except (AuditDataError, SplitError) as exc:
            print(f"split: {exc}", file=sys.stderr)
            return 1

        print(
            f"split: wrote {report['row_count']} rows across "
            f"{report['n_splits']} folds"
        )
        print(f"split sha256: {report['split_sha256']}")
        print(f"assignments: {report['artifacts']['assignments']}")
        print(f"report: {report['artifacts']['report']}")
        return 0

    if args.command == "train":
        try:
            result = train_from_config(args.config)
        except (AuditDataError, SplitError, TrainingError) as exc:
            print(f"train: {exc}", file=sys.stderr)
            return 1

        state = "reused" if result["reused"] else "completed"
        print(f"train: {state} grouped OOF benchmark run {result['run_id']}")
        print(f"run directory: {result['run_dir']}")
        for name, model_metrics in result["metrics"].items():
            print(
                f"  {name}: balanced_accuracy={model_metrics['balanced_accuracy']:.4f}, "
                f"macro_f1={model_metrics['macro_f1']:.4f}"
            )
        return 0

    if args.command == "fit-final":
        try:
            result = fit_final_from_config(args.config)
        except (AuditDataError, TrainingError, InferenceError) as exc:
            print(f"fit-final: {exc}", file=sys.stderr)
            return 1
        state = "reused" if result["reused"] else "fitted"
        print(f"fit-final: {state} model {result['model_id']}")
        print(f"model bundle: {result['model_dir']}")
        return 0

    if args.command == "predict":
        try:
            output_path = args.output or default_submission_output(args.config)
            result = predict_from_artifact(args.model, args.input, output_path)
        except InferenceError as exc:
            print(f"predict: {exc}", file=sys.stderr)
            return 1
        print(f"predict: wrote {result['rows']} rows for model {result['model_id']}")
        print(f"output: {result['output']}")
        return 0

    if args.command == "validate-submission":
        try:
            result = validate_submission_from_config(args.input, args.test, args.config)
        except InferenceError as exc:
            print(f"validate-submission: {exc}", file=sys.stderr)
            return 1
        print(f"validate-submission: {result['status']}")
        for issue in result["issues"]:
            print(f"  - {issue}", file=sys.stderr)
        return 0 if result["status"] == "passed" else 1

    if args.command == "evaluate":
        try:
            result = analyze_oof_run(
                args.run_id,
                args.config,
                candidate_name=args.candidate,
            )
        except AnalysisError as exc:
            print(f"evaluate: {exc}", file=sys.stderr)
            return 1
        summary = result["summary"]
        print(
            f"evaluate: {summary['candidate']} balanced_accuracy="
            f"{summary['balanced_accuracy']:.4f}; errors={summary['error_count']}"
        )
        print(f"error report: {result['report_path']}")
        return 0

    if args.command == "tune":
        try:
            result = tune_from_config(args.config, args.experiment)
        except AnalysisError as exc:
            print(f"tune: {exc}", file=sys.stderr)
            return 1
        print(f"tune: {result['status']} — {result['successful_count']}/{result['candidate_count']} candidates")
        for candidate in result["candidates"][:5]:
            print(
                f"  #{candidate['rank']} {candidate['name']}: "
                f"balanced_accuracy={candidate['balanced_accuracy']:.4f} "
                f"±{candidate['balanced_accuracy_std']:.4f}"
            )
        print(f"summary: {result['summary_path']}")
        return 0 if result["status"] == "passed" else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
