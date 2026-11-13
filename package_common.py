from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def release_check():
    manifest = read(ROOT / 'source_manifest.json')
    for name, expected in manifest['files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Changed release source/configuration: ' + name)
    return manifest


def check_inputs(case, include_analysis=True):
    records = read(ROOT / 'inputs_to_copy.json')['files']
    selected = [r for r in records if r['case'] in (case, 'shared') and (include_analysis or not r['role'].startswith('GN comparison'))]
    missing = [r['destination'] for r in selected if not (ROOT / r['destination']).is_file()]
    if missing:
        raise FileNotFoundError('Required inputs: ' + ', '.join(missing))
    for item in selected:
        path = ROOT / item['destination']
        if path.stat().st_size != item['destination_bytes'] or sha(path) != item['destination_sha256']:
            raise ValueError('Changed input: ' + item['destination'])
    return selected


def resolved_config(case, output, initial=''):
    cfg = read(ROOT / 'config' / (case + '.json'))
    for key in ('data_path', 'index_path', 'checkpoint_path', 'field_manifest_path'):
        if cfg.get(key):
            cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg['result_dir'] = str(Path(output).resolve())
    cfg['initial_model_path'] = str(Path(initial).resolve()) if initial else ''
    return cfg


def load_module(name, path):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
