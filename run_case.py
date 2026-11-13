from pathlib import Path
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from package_common import ROOT, read, write, sha, release_check, check_inputs, resolved_config, load_module


def invoke(code, script, arguments, output, records, phase):
    log = output / 'logs' / (phase + '.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = dict(os.environ, AEM_PACKAGE_ROOT=str(ROOT), PYTHONUTF8='1', MPLBACKEND='Agg')
    command = [sys.executable, '-u', '-X', 'utf8', '-B', str(code / script), *arguments]
    print(json.dumps({'phase': phase, 'log': str(log)}, ensure_ascii=False), flush=True)
    with log.open('w', encoding='utf8') as stream:
        completed = subprocess.run(command, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT)
    records.append({'phase': phase, 'seconds': time.perf_counter() - started, 'exit_code': completed.returncode, 'log': str(log), 'log_sha256': sha(log)})
    write(output / 'phase_timing.json', records)
    if completed.returncode:
        raise RuntimeError(phase + ' failed; retained log: ' + str(log))


def prepare_field_phase(output, phase):
    phase_output = output / phase
    code = phase_output / '_code'
    code.mkdir(parents=True)
    bindings = {}
    for source in (ROOT / 'code' / 'field' / phase).glob('*.py'):
        shutil.copy2(source, code / source.name)
        bindings[source.name] = source.relative_to(ROOT).as_posix()
    source = ROOT / 'code/field/validate_protocol.py'
    shutil.copy2(source, code / source.name)
    bindings[source.name] = source.relative_to(ROOT).as_posix()
    for source in (ROOT / 'data/orkdalen/metadata').iterdir():
        if source.is_file():
            shutil.copy2(source, code / source.name)
    point = 'cold_DCT_MAP180' if phase == 'cold180' else 'native_MAP240'
    config = resolved_config('orkdalen', phase_output / 'joint_VI160', phase_output / point / 'selected_model_log10.npy')
    protocol = {'status': 'portable_runtime_bound', 'phase': phase, 'output': str(phase_output), 'config': config,
                'runtime_source_bindings': bindings, 'source_manifest_sha256': sha(ROOT / 'source_manifest.json')}
    protocol.update(read(ROOT / 'config' / ('field_' + phase + '_algorithm.json')))
    if phase == 'native240':
        prefix = output / 'cold180' / 'cold_DCT_MAP180'
        completed = read(prefix / 'completed_MAP.json')
        protocol['native_refinement'].update(
            source_cold_MAP_checkpoint=str(prefix / 'best.pt'),
            source_cold_MAP_checkpoint_sha256=sha(prefix / 'best.pt'),
            source_cold_MAP_model=str(prefix / 'selected_model_log10.npy'),
            source_cold_MAP_model_sha256=sha(prefix / 'selected_model_log10.npy'),
            source_cold_MAP_completion=str(prefix / 'completed_MAP.json'),
            source_cold_protocol=str(output / 'cold180/_code/execution_protocol.json'),
            inherited_cold_MAP_seconds=completed['elapsed_seconds_including_load'])
    write(code / 'execution_protocol.json', protocol)
    write(code / 'config_omega50.json', config)
    return code, protocol


def run_field(output):
    phases = []
    code, protocol = prepare_field_phase(output, 'cold180')
    invoke(code, 'probe_stage3.py', ['--config', str(code / 'config_omega50.json'), '--gpu-granted'], output, phases, 'cold180_resource_probe')
    invoke(code, 'run_cold_map.py', ['--execute', '--gpu-granted'], output, phases, 'cold180_point_fit')
    code, protocol = prepare_field_phase(output, 'native240')
    invoke(code, 'probe_stage3.py', ['--config', str(code / 'config_omega50.json'), '--gpu-granted'], output, phases, 'native240_resource_probe')
    invoke(code, 'run_cold_map.py', ['--execute', '--gpu-granted'], output, phases, 'native240_point_fit')
    invoke(code, 'AEM_Variational_Bayes_3D_FD_v8.py', ['--config', str(code / 'config_omega50.json')], output, phases, 'full_joint_VI160')
    invoke(code, 'export_result.py', ['--execute', '--gpu-granted'], output, phases, 'posterior1024_quantiles256_predictive8')
    recipe = Path(protocol['config']['result_dir']) / 'posterior/sampling_recipe_cv_v3.json'
    invoke(code, 'rebuild_cv_draws.py', [str(recipe), '--execute', '--gpu-granted'], output, phases, 'sampling_hash_replay')
    invoke(code, 'audit_joint_distribution.py', ['--omega', '50', '--gpu-granted'], output, phases, 'independent_density_objective_CV_audit')
    return {'run': protocol['config']['result_dir'], 'phases': phases}


def run_synthetic(case, output, with_analysis):
    code = ROOT / 'code/synthetic'
    core = load_module('portable_' + case, code / (case + '.py'))
    core.ROOT = code
    cfg = core.Config(**resolved_config(case, output / 'run'))
    core.validate_config(cfg)
    started = time.perf_counter()
    core.train(cfg)
    execution = {'status': 'completed', 'case': case, 'inversion_wall_seconds': time.perf_counter() - started,
                 'truth_used_in_target_or_checkpoint_selection': False, 'source_manifest_sha256': sha(ROOT / 'source_manifest.json'),
                 'selected_checkpoint_sha256': sha(output / 'run/best.pt')}
    write(output / 'execution.json', execution)
    cfg.truth_path = str(ROOT / 'data' / case / 'truth.rho')
    execution['truth_assessment'] = core.assess_result(output / 'run', cfg)
    write(output / 'execution.json', execution)
    if with_analysis:
        analysis = load_module('portable_structural_analysis', code / 'postprocess.py')
        analysis.BASE = output
        analysis.PLAN = output / 'analysis_plan.json'
        analysis.CASES = {case: {'case': code, 'run': output / 'run', 'execution': output / 'execution.json',
                                'core': code / (case + '.py'), 'threshold': 50.0 if case == 'staggered' else 31.622776601683793,
                                'truth': ROOT / 'data' / case / 'truth.rho', 'GN': str(ROOT / 'data' / case / 'gn.rho')}}
        analysis.prepare()
        analysis.analyze(case)
    return {'run': str(output / 'run'), 'structural_analysis': with_analysis}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['staggered', 'ovid', 'orkdalen'], required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--skip-analysis', action='store_true')
    args = parser.parse_args()
    release_check()
    entries = check_inputs(args.case, not args.skip_analysis)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output = (args.output or ROOT / 'runs' / (args.case + '_' + stamp)).resolve()
    if not output.is_relative_to((ROOT / 'runs').resolve()) or output == (ROOT / 'runs').resolve():
        raise ValueError('Choose a new subdirectory of this package runs directory')
    if output.exists():
        raise FileExistsError('Existing output is preserved: ' + str(output))
    config = resolved_config(args.case, output / 'run')
    print(json.dumps({'status': 'input_and_source_checks_passed', 'case': args.case, 'files_checked': len(entries),
                      'output': str(output), 'execute': args.execute, 'model_shape': [config['ny'], config['nx'], config['nz']]}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('The full inversion requires a CUDA GPU; input checks do not require a GPU')
    os.environ['AEM_PACKAGE_ROOT'] = str(ROOT)
    output.mkdir(parents=True)
    started = time.perf_counter()
    record = {'status': 'running', 'case': args.case, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_manifest_sha256': sha(ROOT / 'source_manifest.json')}
    write(output / 'reproduction.json', record)
    try:
        record['result'] = run_field(output) if args.case == 'orkdalen' else run_synthetic(args.case, output, not args.skip_analysis)
        record['status'] = 'completed_new_reproduction_not_original_published_run'
    except BaseException as exc:
        record.update(status='failed_outputs_retained', error=str(exc))
        raise
    finally:
        record['elapsed_seconds'] = time.perf_counter() - started
        write(output / 'reproduction.json', record)


if __name__ == '__main__':
    main()
