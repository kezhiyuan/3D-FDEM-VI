from pathlib import Path
from dataclasses import asdict
import os
import sys
import numpy as np

PACKAGE = Path(os.environ['AEM_PACKAGE_ROOT']).resolve()
sys.path.insert(0, str(PACKAGE))
from package_common import read, sha, release_check, resolved_config

HERE = Path(__file__).resolve().parent


def require_frozen_protocol():
    release_check()
    p = read(HERE / 'execution_protocol.json')
    if p['source_manifest_sha256'] != sha(PACKAGE / 'source_manifest.json'):
        raise ValueError('Runtime protocol does not match this source release')
    if p['status'] != 'portable_runtime_bound' or p['phase'] not in ('cold180', 'native240'):
        raise ValueError('Invalid portable field protocol')
    output = Path(p['output']).resolve()
    if not output.is_relative_to((PACKAGE / 'runs').resolve()):
        raise ValueError('Output must be a new directory within package runs')
    point = 'cold_DCT_MAP180' if p['phase'] == 'cold180' else 'native_MAP240'
    expected = resolved_config('orkdalen', output / 'joint_VI160', output / point / 'selected_model_log10.npy')
    if p['config'] != expected:
        raise ValueError('Changed numerical settings or runtime paths')
    algorithm = read(PACKAGE / 'config' / ('field_' + p['phase'] + '_algorithm.json'))
    for key, value in algorithm.items():
        if key == 'native_refinement':
            if any(p[key].get(k) != v for k, v in value.items()):
                raise ValueError('Changed native refinement algorithm')
        elif p.get(key) != value:
            raise ValueError('Changed algorithm field: ' + key)
    expected_sources = read(PACKAGE / 'source_manifest.json')['files']
    for name, original in p['runtime_source_bindings'].items():
        if original not in expected_sources or sha(HERE / name) != expected_sources[original]:
            raise ValueError('Changed staged numerical source: ' + name)
    entries = read(PACKAGE / 'inputs_to_copy.json')['files']
    for item in entries:
        if item['case'] in ('orkdalen', 'shared') and not item['destination'].endswith(('.rho', '.imp')):
            if sha(PACKAGE / item['destination']) != item['destination_sha256']:
                raise ValueError('Changed input: ' + item['destination'])
            if '/metadata/' in item['destination'] and sha(HERE / Path(item['destination']).name) != item['destination_sha256']:
                raise ValueError('Changed staged field metadata')
    if p['phase'] == 'native240':
        options = p['native_refinement']
        prefix = Path(options['source_cold_MAP_completion']).parent
        if prefix.resolve() != output.parent / 'cold180' / 'cold_DCT_MAP180':
            raise ValueError('Point initializer must come from this fresh cold180 run')
        completed = read(prefix / 'completed_MAP.json')
        if not (completed['status'] == 'completed' and completed['cold_uniform100'] and not completed['truth_or_GN_used'] and completed['selected_step'] == 180):
            raise ValueError('Cold prefix is incomplete or not the declared step180')
        for name, key in [('best.pt', 'selected_checkpoint_sha256'), ('selected_model_log10.npy', 'selected_model_sha256')]:
            if sha(prefix / name) != completed[key]:
                raise ValueError('Cold prefix output changed: ' + name)
        if sha(options['source_cold_protocol']) != completed['protocol_sha256']:
            raise ValueError('Cold prefix protocol changed')
    return p


def validate_training_inputs(cfg, require_warmstart=True):
    p = require_frozen_protocol()
    if asdict(cfg) != p['config']:
        raise ValueError('Training configuration differs from bound protocol')
    observations = np.loadtxt(cfg.data_path, ndmin=2)
    if observations.shape != (3949, 14) or not np.array_equal(observations[:, 0], np.arange(3949)):
        raise ValueError('Require all 3949 original station records once in order')
    if cfg.truth_path or cfg.warmstart_flow_checkpoint_path or cfg.map_steps or cfg.map_refine_steps:
        raise ValueError('Unexpected external initialization or MAP updates')
    if not require_warmstart:
        return {'status': 'pass_before_probe', 'station_count': 3949}
    target = Path(cfg.initial_model_path)
    completed = read(target.parent / 'completed_MAP.json')
    if not (completed['status'] == 'completed' and completed['cold_uniform100'] and not completed['truth_or_GN_used'] and completed['station_count'] == 3949 and completed['selected_model_sha256'] == sha(target) and completed['protocol_sha256'] == sha(HERE / 'execution_protocol.json')):
        raise ValueError('Selected point initialization is not bound to this completed phase')
    if p['phase'] == 'native240' and (completed['additional_native_MAP_steps'] != 240 or completed['source_cold_MAP_checkpoint_sha256'] != p['native_refinement']['source_cold_MAP_checkpoint_sha256']):
        raise ValueError('Native refinement does not descend from the fresh cold180 prefix')
    return {'status': 'pass', 'station_count': 3949, 'protocol_sha256': sha(HERE / 'execution_protocol.json')}
