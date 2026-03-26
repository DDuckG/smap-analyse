from pathlib import Path
import yaml
from smap.ontology.models import OntologyRegistry

def _read_yaml(path):
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if payload is None:
        raise ValueError(f'Ontology file is empty: {path}')
    return payload

def load_ontology(path):
    return OntologyRegistry.model_validate(_read_yaml(path))
