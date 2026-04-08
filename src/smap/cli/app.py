from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from smap.core.settings import Settings, get_settings
from smap.pipeline import run_pipeline
from smap.review.migrations import get_database_status, upgrade_database
from smap.runtime.doctor import run_runtime_doctor
from smap.validation.service import validate_batch


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    updates: dict[str, object] = {}
    if getattr(args, "data_dir", None):
        updates["data_dir"] = Path(args.data_dir)
    if getattr(args, "db_url", None):
        updates["db_url"] = args.db_url
    if getattr(args, "domain_id", None):
        updates["domain_id"] = args.domain_id
    if getattr(args, "domain_ontology", None):
        updates["domain_ontology_path"] = Path(args.domain_ontology)
    if updates:
        settings = settings.model_copy(update=updates)
    settings.ensure_directories()
    return settings


def _echo_json(payload: object) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
        return
    print(text)


def _progress_callback(prefix: str):
    def callback(stage_name: str, event: str, elapsed_seconds: float | None) -> None:
        if event == "start":
            print(f"[{prefix}] {stage_name}...", flush=True)
            return
        print(f"[{prefix}] {stage_name} done in {elapsed_seconds or 0.0:.2f}s", flush=True)

    return callback


def cmd_list_domains(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    payload = [{"name": path.name, "path": str(path)} for path in settings.available_domain_ontology_paths()]
    _echo_json(payload)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    _echo_json(run_runtime_doctor(settings).model_dump(mode="json"))
    return 0


def cmd_db_upgrade(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    upgrade_database(settings)
    status = get_database_status(settings)
    _echo_json(
        {
            "current_revision": status.current_revision,
            "head_revision": status.head_revision,
            "is_ready": status.is_ready,
        }
    )
    return 0


def cmd_db_current(args: argparse.Namespace) -> int:
    status = get_database_status(_settings_from_args(args))
    _echo_json(
        {
            "current_revision": status.current_revision,
            "head_revision": status.head_revision,
            "is_ready": status.is_ready,
            "has_legacy_tables_without_alembic": status.has_legacy_tables_without_alembic,
        }
    )
    return 0


def cmd_validate_batch(args: argparse.Namespace) -> int:
    report = validate_batch(Path(args.input_path))
    if args.output_json:
        Path(args.output_json).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    _echo_json(report.model_dump(mode="json"))
    return 0


def cmd_run_pipeline(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    result = run_pipeline(
        Path(args.input_path),
        settings=settings,
        progress_callback=_progress_callback("pipeline"),
    )
    if args.output_json:
        Path(args.output_json).write_text(result.model_dump_json(indent=2), encoding="utf-8")
    _echo_json(result.model_dump(mode="json"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smap",
        description="SMAP clean ONNX CPU pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data-dir", help="Override the runtime data root. Defaults to ./var/data.")
    shared.add_argument("--db-url", help="Override the metadata DB URL. Defaults to sqlite:///./var/app.db.")
    shared.add_argument("--domain-id", help="Select a bundled domain pack such as cosmetics_vn, beer_vn, or blockchain_vn.")
    shared.add_argument("--domain-ontology", help="Force a specific domain ontology YAML path.")

    doctor = subparsers.add_parser(
        "doctor",
        parents=[shared],
        help="Report ONNX CPU runtime readiness.",
    )
    doctor.set_defaults(handler=cmd_doctor)

    list_domains = subparsers.add_parser(
        "list-domains",
        parents=[shared],
        help="List bundled domain ontologies.",
    )
    list_domains.set_defaults(handler=cmd_list_domains)

    db_upgrade = subparsers.add_parser(
        "db-upgrade",
        parents=[shared],
        help="Initialize or upgrade the metadata DB.",
    )
    db_upgrade.set_defaults(handler=cmd_db_upgrade)

    db_current = subparsers.add_parser(
        "db-current",
        parents=[shared],
        help="Show the current DB revision state.",
    )
    db_current.set_defaults(handler=cmd_db_current)

    validate = subparsers.add_parser(
        "validate-batch",
        help="Validate a JSONL, ZIP, or directory batch.",
    )
    validate.add_argument("input_path")
    validate.add_argument("--output-json", help="Write the validation report to a JSON file.")
    validate.set_defaults(handler=cmd_validate_batch)

    run_cmd = subparsers.add_parser(
        "run-pipeline",
        parents=[shared],
        help="Run the full pipeline on the ONNX CPU runtime.",
    )
    run_cmd.add_argument("input_path")
    run_cmd.add_argument("--output-json", help="Write the pipeline summary to a JSON file.")
    run_cmd.set_defaults(handler=cmd_run_pipeline)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
