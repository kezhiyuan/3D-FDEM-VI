from pathlib import Path
import argparse
import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import tokenize
from package_common import ROOT, read, release_check, check_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', action='store_true')
    parser.add_argument('--imports', action='store_true')
    parser.add_argument('--protocol', action='store_true')
    args = parser.parse_args()
    manifest = release_check()
    failures = []
    code_count = 0
    for rel in manifest['files']:
        if not rel.endswith('.py'):
            continue
        path = ROOT / rel
        text = path.read_text('utf8')
        tree = ast.parse(text)
        compile(tree, str(path), 'exec')
        if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(text).readline)):
            failures.append('Comment: ' + rel)
        if any(ast.get_docstring(n) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))):
            failures.append('Docstring: ' + rel)
        code_count += 1
    checked_inputs = {}
    if args.inputs:
        for case in ['staggered', 'ovid', 'orkdalen']:
            checked_inputs[case] = len(check_inputs(case))
    if args.imports:
        targets = [('code/synthetic/staggered.py', 'staggered'), ('code/synthetic/ovid.py', 'ovid'),
                   ('code/field/cold180/AEM_Variational_Bayes_3D_FD_v8.py', 'orkdalen'),
                   ('code/field/native240/AEM_Variational_Bayes_3D_FD_v8.py', 'orkdalen')]
        for rel, case in targets:
            command = 'from package_common import ROOT,load_module,read; m=load_module("cpu_import",ROOT/' + repr(rel) + '); c=m.Config(**read(ROOT/"config"/' + repr(case + '.json') + ')); print(type(c).__name__,c.ny,c.nx,c.nz)'
            env = dict(os.environ, AEM_PACKAGE_ROOT=str(ROOT), PYTHONUTF8='1', MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='')
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', command], cwd=ROOT, env=env, text=True, encoding='utf8', capture_output=True)
            if result.returncode:
                failures.append({'import': rel, 'error': result.stderr})
    protocol_checks = None
    if args.protocol:
        runs = ROOT / 'runs'
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='cpu_protocol_', dir=runs) as directory:
            command = '\n'.join([
                'from pathlib import Path',
                'from run_case import prepare_field_phase',
                'from package_common import load_module',
                'import json',
                'code, protocol = prepare_field_phase(Path(' + repr(directory) + '), "cold180")',
                'guard = load_module("cpu_field_guard", code / "validate_protocol.py")',
                'core = load_module("cpu_field_core", code / "AEM_Variational_Bayes_3D_FD_v8.py")',
                'cfg = core.Config(**protocol["config"])',
                'record = guard.validate_training_inputs(cfg, require_warmstart=False)',
                'cfg.data_weight += 1.0',
                'try:',
                '    guard.validate_training_inputs(cfg, require_warmstart=False)',
                'except ValueError:',
                '    record["changed_configuration_rejected"] = True',
                'else:',
                '    raise AssertionError("Changed numerical configuration was accepted")',
                'record["native240_runtime_checked"] = False',
                'print(json.dumps(record))',
            ])
            env = dict(os.environ, AEM_PACKAGE_ROOT=str(ROOT), PYTHONUTF8='1', MPLBACKEND='Agg', CUDA_VISIBLE_DEVICES='')
            result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', command], cwd=ROOT, env=env, text=True, encoding='utf8', capture_output=True)
            if result.returncode:
                failures.append({'protocol': 'cold180', 'error': result.stderr})
            else:
                protocol_checks = json.loads(result.stdout.strip().splitlines()[-1])
    report = {'status': 'pass' if not failures else 'fail', 'python_sources': code_count,
              'release_files': len(manifest['files']), 'inputs_checked': checked_inputs,
              'CPU_imports_checked': args.imports, 'GPU_used': False, 'inversion_or_sampling_run': False,
              'CPU_field_protocol': protocol_checks,
              'failures': failures}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
