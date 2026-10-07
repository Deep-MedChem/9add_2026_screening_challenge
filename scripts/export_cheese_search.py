#!/usr/bin/env python3
"""Validate and package a completed CHEESE search as gzipped CSVs in a ZIP."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from search_cheese_real import export_csv, read_queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    config = manifest['config']
    status = json.loads((args.run / 'run_status.json').read_text())
    if status['phase'] != 'complete' or status.get('errors'):
        raise SystemExit('Search has not completed successfully')
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != config['input_sha256']:
        raise SystemExit('Input differs from the search manifest')
    queries = read_queries(args.input)
    if queries != json.loads((args.run / 'queries.json').read_text()):
        raise SystemExit('Query snapshot differs from input')
    expected = len(queries) * config['n_neighbors']
    for metric in config['methods']:
        count = export_csv(args.run, metric, queries, config['n_neighbors'])
        if count != expected:
            raise SystemExit(f'{metric}: expected {expected} rows, found {count}')
    args.package.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.package.parent) as temp:
        package = Path(temp) / args.package.name
        (package / 'data').mkdir(parents=True)
        shutil.copyfile(args.input, package / 'data' / args.input.name)
        for metric in config['methods']:
            name = f"{metric}_top{config['n_neighbors']}.csv"
            with (args.run / name).open('rb') as source, gzip.open(package / (name + '.gz'), 'wb') as dest:
                shutil.copyfileobj(source, dest)
        info = f"""# CHEESE search: {args.input.stem}

Input: `data/{args.input.name}` ({len(queries)} valid SMILES and unique IDs).
Input SHA-256: `{config['input_sha256']}`.
Completed: {status['finished_at']}.

Database: **{config['database']}**. Search quality: **{config['quality']}**.
Methods: {json.dumps(config['methods'])}.
Neighbors: **{config['n_neighbors']} per query per metric**, **{expected:,} rows per CSV**.
API: {config['api_url']}.
Database metadata: {json.dumps(manifest['database_metadata'])}.

No property filters; sim_th=0, d_min=0, d_max=1. Property and synthon retrieval disabled.
Jobs API with {config['page_size']} hits per page; one sequential worker per metric.
The hosted deployment does not expose a model revision or dated database release.
Accurate search is approximate retrieval, not an exhaustive nearest-neighbor guarantee.

CSV columns: `{','.join(config['columns'])}`.
Ranks are one-based within each query, sorted by full-precision cosine similarity.
CSV scores have three decimal places. IDs and SMILES are preserved as returned.
The synthetic Query Molecule row is excluded; result IDs are unique within each query.
Hits shared between queries are retained.

The CSVs are gzip-compressed; the original input is included unchanged.
Checkpoints, full-precision scores, manifest, API schema, and logs remain in
`{args.run}` in the working repository and are not included in this package.
"""
        (package / 'INFO.md').write_text(info)
        archive = Path(temp) / (args.package.name + '.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(package.rglob('*')):
                if path.is_file():
                    bundle.write(path, path.relative_to(Path(temp)))
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise SystemExit('ZIP integrity check failed')
        shutil.copytree(package, args.package, dirs_exist_ok=True)
        archive.replace(args.package.with_suffix('.zip'))
    print(f'Export complete: {args.package.with_suffix(".zip")} ({expected:,} rows per metric)', flush=True)


if __name__ == '__main__':
    main()
