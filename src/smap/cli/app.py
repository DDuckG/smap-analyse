import argparse
import json
from pathlib import Path
from smap.core.settings import get_settings
from smap.pipeline import run_pipeline
from smap.review.migrations import get_database_status, upgrade_database
from smap.runtime.doctor import run_runtime_doctor
from smap.validation.service import validate_batch

def _settings_from_args(args):
    settings = get_settings()
    updates = {}
    if getattr(args, 'data_dir', None):
        updates['data_dir'] = Path(args.data_dir)
    if getattr(args, 'db_url', None):
        updates['db_url'] = args.db_url
    if getattr(args, 'domain_ontology', None):
        updates['domain_ontology_path'] = Path(args.domain_ontology)
    if getattr(args, 'degraded_fallback', False):
        updates['intelligence'] = settings.intelligence.model_copy(update={'enable_optional_ml_providers': False, 'semantic_assist_enabled': False, 'semantic_hypothesis_rerank_enabled': False, 'semantic_corroboration_enabled': False, 'entity_embedding_rerank_enabled': False, 'topic_label_rerank_enabled': False})
    if updates:
        settings = settings.model_copy(update=updates)
    settings.ensure_directories()
    return settings

def _echo_json(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

def _progress_callback(prefix):

    def callback(stage_name, event, elapsed_seconds):
        if event == 'start':
            print(f'[{prefix}] {stage_name}...', flush=True)
            return
        print(f'[{prefix}] {stage_name} done in {elapsed_seconds or 0.0:.2f}s', flush=True)
    return callback

def cmd_list_domains(args):
    settings = _settings_from_args(args)
    payload = [{'name': path.name, 'path': str(path)} for path in settings.available_domain_ontology_paths()]
    _echo_json(payload)
    return 0

def cmd_doctor(args):
    settings = _settings_from_args(args)
    _echo_json(run_runtime_doctor(settings).model_dump(mode='json'))
    return 0

def cmd_db_upgrade(args):
    settings = _settings_from_args(args)
    upgrade_database(settings)
    status = get_database_status(settings)
    _echo_json({'current_revision': status.current_revision, 'head_revision': status.head_revision, 'is_ready': status.is_ready})
    return 0

def cmd_validate_batch(args):
    report = validate_batch(Path(args.input_path))
    if args.output_json:
        Path(args.output_json).write_text(report.model_dump_json(indent=2), encoding='utf-8')
    _echo_json(report.model_dump(mode='json'))
    return 0

def cmd_run_pipeline(args):
    settings = _settings_from_args(args)
    result = run_pipeline(Path(args.input_path), settings=settings, progress_callback=_progress_callback('pipeline'))
    if args.output_json:
        Path(args.output_json).write_text(result.model_dump_json(indent=2), encoding='utf-8')
    _echo_json(result.model_dump(mode='json'))
    return 0

def build_parser():
    parser = argparse.ArgumentParser(prog='smap', description='SMAP core pipeline handoff CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument('--data-dir', help='Override the runtime data root. Defaults to ./var/data.')
    shared.add_argument('--db-url', help='Override the metadata DB URL. Defaults to sqlite:///./var/app.db.')
    shared.add_argument('--domain-ontology', help='Force a specific domain ontology YAML path.')
    doctor = subparsers.add_parser('doctor', parents=[shared], help='Check runtime readiness for the ML path.')
    doctor.set_defaults(handler=cmd_doctor)
    list_domains = subparsers.add_parser('list-domains', parents=[shared], help='List bundled domain ontologies.')
    list_domains.set_defaults(handler=cmd_list_domains)
    db_upgrade = subparsers.add_parser('db-upgrade', parents=[shared], help='Initialize or upgrade the metadata DB.')
    db_upgrade.set_defaults(handler=cmd_db_upgrade)
    validate = subparsers.add_parser('validate-batch', help='Validate a JSONL, ZIP, or directory batch.')
    validate.add_argument('input_path')
    validate.add_argument('--output-json', help='Write the validation report to a JSON file.')
    validate.set_defaults(handler=cmd_validate_batch)
    run_cmd = subparsers.add_parser('run-pipeline', parents=[shared], help='Run the full core pipeline.')
    run_cmd.add_argument('input_path')
    run_cmd.add_argument('--output-json', help='Write the pipeline summary to a JSON file.')
    run_cmd.add_argument('--degraded-fallback', action='store_true', help='Disable optional ML providers and force deterministic fallback mode.')
    run_cmd.set_defaults(handler=cmd_run_pipeline)
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
if __name__ == '__main__':
    raise SystemExit(main())
