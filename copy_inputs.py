from pathlib import Path
import argparse
import json
import re
import shutil
from package_common import ROOT, read, sha


def normalize(value, source_root):
    if isinstance(value, dict):
        return {normalize(k, source_root): normalize(v, source_root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, source_root) for v in value]
    if isinstance(value, str) and re.match(r'^[A-Za-z]:[\\/]', value):
        try:
            return 'archive-relative/' + Path(value).resolve().relative_to(source_root).as_posix()
        except ValueError:
            marker = '3D_FDEM_SelfSupervised_VI/'
            normalized = value.replace('\\', '/')
            if marker in normalized:
                return 'archive-relative/' + normalized.split(marker, 1)[1]
            return 'external-archive/' + normalized.rsplit('/', 1)[-1]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--case', choices=['staggered', 'ovid', 'orkdalen', 'all'], default='all')
    parser.add_argument('--include-large', action='store_true')
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    copied = []
    for item in read(ROOT / 'inputs_to_copy.json')['files']:
        if args.case != 'all' and item['case'] not in (args.case, 'shared'):
            continue
        if item['large_artifact'] and not args.include_large:
            continue
        source = (source_root / item['source']).resolve()
        target = (ROOT / item['destination']).resolve()
        if not source.is_relative_to(source_root) or not target.is_relative_to(ROOT):
            raise ValueError('Path escapes its root')
        if sha(source) != item['source_sha256']:
            raise ValueError('Source differs from the audited input: ' + item['source'])
        if target.exists():
            if sha(target) != item['destination_sha256']:
                previous = target.read_bytes()
                import hashlib
                normalized = previous.replace(b'\r\n', b'\n')
                if item['transformation'] == 'normalize_json_provenance_paths' and hashlib.sha256(normalized).hexdigest() == item['destination_sha256']:
                    target.write_bytes(normalized)
                else:
                    raise FileExistsError('Different existing input: ' + str(target))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item['transformation'] == 'normalize_json_provenance_paths':
            obj = normalize(read(source), source_root)
            target.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf8'))
        else:
            shutil.copy2(source, target)
        if sha(target) != item['destination_sha256']:
            raise ValueError('Copied input digest mismatch: ' + item['destination'])
        copied.append(item['destination'])
    print(json.dumps({'copied': copied, 'large_files_included': args.include_large}, ensure_ascii=False))


if __name__ == '__main__':
    main()
