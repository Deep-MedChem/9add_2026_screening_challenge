#!/usr/bin/env python3
"""Resumable CHEESE Jobs API searches, one worker per similarity metric."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import requests

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"espsim": "espsim_electrostatic", "shapesim": "espsim_shape"}
COLUMNS = ["query_smiles", "query_id", "result_smiles", "result_rank", "result_id", "cosine_similarity"]
SECRET = "op://Developer/deuoei4pnpgj72bh6fd6ecbe4y/credential"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def read_queries(path):
    from rdkit import Chem
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2 or Chem.MolFromSmiles(fields[0]) is None:
            raise ValueError(f"Invalid SMILES/ID row {number}")
        rows.append({"smiles": fields[0], "id": fields[1]})
    if not rows or len({q['id'] for q in rows}) != len(rows):
        raise ValueError("Input must contain unique, nonempty query IDs")
    return rows


def request(session, base, method, endpoint, *, submitting=False, **kwargs):
    for attempt in range(8):
        try:
            response = session.request(method, base + endpoint, timeout=(15, 120), **kwargs)
        except requests.RequestException:
            if submitting or attempt == 7:
                raise RuntimeError(f"{endpoint}: network error; submission is not automatically retried") from None
            time.sleep(min(60, 2 ** attempt))
            continue
        if response.status_code == 429 or (response.status_code >= 500 and not submitting):
            if attempt < 7:
                try:
                    delay = float(response.headers.get("Retry-After", 2 ** attempt))
                except ValueError:
                    delay = 2 ** attempt
                print(f"{now()} {endpoint} HTTP {response.status_code}; retry in {delay}s", flush=True)
                time.sleep(max(1, min(300, delay)))
                continue
        if not response.ok:
            # Do not log request headers or response bodies containing credentials.
            raise RuntimeError(f"{endpoint}: HTTP {response.status_code}")
        return response.json()
    raise RuntimeError(f"{endpoint}: retry budget exhausted")


def parse_hits(page):
    arrays = [page.get(k) for k in ("id", "smiles", "similarity")]
    if not all(isinstance(a, list) for a in arrays) or len({len(a) for a in arrays}) != 1:
        raise ValueError("Malformed result page: id/smiles/similarity lengths differ")
    hits = []
    for ident, smiles, score in zip(*arrays):
        if ident == "Query Molecule":
            continue
        score = float(score)
        if not ident or not smiles or not math.isfinite(score) or not -1.00001 <= score <= 1.00001:
            raise ValueError("Invalid hit ID, SMILES, or cosine similarity")
        hits.append({"id": str(ident), "smiles": str(smiles), "score": score})
    return hits


def export_csv(out, metric, queries, n):
    target = out / f"{metric}_top{n}.csv"
    temp = target.with_suffix(".csv.tmp")
    count = 0
    with temp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for i, query in enumerate(queries):
            path = out / "checkpoints" / metric / f"{i:03d}.json"
            if not path.exists():
                continue
            checkpoint = json.loads(path.read_text())
            if not checkpoint.get("complete"):
                continue
            hits = checkpoint["hits"]
            if len(hits) != n or len({h['id'] for h in hits}) != n:
                raise ValueError(f"Invalid completed checkpoint: {path}")
            for rank, hit in enumerate(hits, 1):
                writer.writerow([query['smiles'], query['id'], hit['smiles'], rank, hit['id'], f"{hit['score']:.3f}"])
                count += 1
    temp.replace(target)
    return count


def worker(args, metric, search_type, queries, key):
    out = args.out
    folder = out / "checkpoints" / metric
    folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"X-API-Key": key, "accept": "application/json"})
    started = time.time()
    completed = export_csv(out, metric, queries, args.n) // args.n
    initial_completed = completed
    status = {"metric": metric, "completed_queries": completed, "total_queries": len(queries)}

    def progress(phase, **extra):
        status.update(extra, phase=phase, updated_at=now(), completed_queries=completed,
                      csv_rows=completed * args.n, elapsed_seconds=round(time.time() - started))
        done_this_run = completed - initial_completed
        status['eta_seconds'] = round((time.time() - started) / done_this_run * (len(queries) - completed)) if done_this_run else None
        save(out / f"{metric}_status.json", status)
        print(f"{now()} {metric} {completed}/{len(queries)} {phase} query={status.get('query_id', '-')} hits={status.get('hits', 0)}", flush=True)

    try:
        for i, query in enumerate(queries):
            path = folder / f"{i:03d}.json"
            state = json.loads(path.read_text()) if path.exists() else {"query": query, "hits": [], "next_page": 0}
            if state['query'] != query:
                raise ValueError("Checkpoint query mismatch")
            if state.get("complete"):
                continue
            progress("starting", query_id=query['id'], hits=len(state['hits']))
            if not state.get("job_name"):
                if state.get("submission_pending"):
                    raise RuntimeError("Previous submission outcome unknown; reconcile job before retrying")
                state.update(submission_pending=True, submitted_at=now())
                save(path, state)
                payload = request(session, args.api_url, "GET", "/submit_molsearch", submitting=True, params={
                    "search_input": query['smiles'], "search_type": search_type,
                    "search_quality": args.quality, "db_names": args.database,
                    "n_neighbors": args.n, "include_properties": "false"})
                job_name = payload if isinstance(payload, str) else payload.get('job_name', payload.get('job_id'))
                if not isinstance(job_name, str) or not job_name:
                    raise ValueError("Unexpected job submission response")
                state.update(job_name=job_name, submission_pending=False)
                save(path, state)
            job = state['job_name']
            wait_started = time.time()
            while True:
                payload = request(session, args.api_url, "GET", "/job_status", params={"job_name": job})
                remote = payload if isinstance(payload, str) else payload.get('status', payload.get('state'))
                remote = str(remote).upper()
                progress("waiting", job_name=job, remote_status=remote)
                if remote == 'SUCCESS':
                    break
                if remote in {'FAILURE', 'FAILED', 'ERROR', 'REVOKED', 'CANCELLED'}:
                    raise RuntimeError(f"Remote job {job}: {remote}")
                if time.time() - wait_started > args.max_wait:
                    raise TimeoutError(f"Remote job {job} exceeded wait limit; rerun to resume polling")
                time.sleep(args.poll)
            seen = {hit['id'] for hit in state['hits']}
            while len(state['hits']) < args.n:
                page_num = state['next_page']
                page = request(session, args.api_url, "POST", "/get_molsearch_page", params={
                    "job_name": job, "db_name": args.database, "page_size": args.page_size,
                    "page_num": page_num, "include_properties": "false", "include_synthons": "false",
                    "sim_th": 0, "d_min": 0, "d_max": 1}, json={"prop_ranges": {}})
                hits = parse_hits(page)
                added = 0
                for hit in hits:
                    if hit['id'] in seen:
                        continue
                    state['hits'].append(hit)
                    seen.add(hit['id'])
                    added += 1
                if not added:
                    raise RuntimeError(f"Results exhausted/repeated at page {page_num}: {len(seen)}/{args.n} unique hits")
                state['next_page'] += 1
                save(path, state)
                progress("fetching", hits=len(state['hits']), page=page_num)
            # Preserve full precision for ordering; round only the CSV display.
            state['hits'] = sorted(state['hits'], key=lambda h: -h['score'])[:args.n]
            state.update(complete=True, completed_at=now())
            save(path, state)
            completed = export_csv(out, metric, queries, args.n) // args.n
            progress("query_complete", hits=args.n)
        progress("complete")
    except Exception as exc:
        progress("failed", error=str(exc))
        raise
    finally:
        session.close()


def show_status(out):
    for metric in METHODS:
        path = out / f"{metric}_status.json"
        if path.exists():
            print(json.dumps(json.loads(path.read_text()), indent=2))
        else:
            print(f"{metric}: not started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/chembl33_all_r0_top100_sep2026.smi')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    parser.add_argument('--api-url', default='https://api.cheese.deepmedchem.com')
    parser.add_argument('--database', default='ENAMINE-REAL')
    parser.add_argument('--quality', default='accurate', choices=['fast', 'accurate', 'very_accurate'])
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--page-size', type=int, default=99)
    parser.add_argument('--poll', type=float, default=5)
    parser.add_argument('--max-wait', type=float, default=7200)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        show_status(args.out)
        return
    if not 1 <= args.n <= 100000 or not 1 <= args.page_size <= 99:
        parser.error('n must be 1..100000 and page-size 1..99')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'run.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another search is already using this output directory')
    queries = read_queries(args.input)
    config = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "queries": len(queries), "api_url": args.api_url, "database": args.database,
              "quality": args.quality, "n_neighbors": args.n, "page_size": args.page_size,
              "methods": METHODS, "sim_th": 0, "property_filters": {}, "columns": COLUMNS}
    manifest_path = args.out / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config:
            raise SystemExit('Settings/input changed: use a new output directory')
    else:
        manifest = {'created_at': now(), 'config': config}
    key = os.environ.get('CHEESE_API_KEY')
    if not key:
        proc = subprocess.run(['with-keyring', '--profile', 'personal', 'op', 'read', SECRET], capture_output=True, text=True)
        if proc.returncode or not proc.stdout.strip():
            raise SystemExit('Could not retrieve the requested CHEESE credential from 1Password')
        key = proc.stdout.strip()
    with requests.Session() as session:
        session.headers['X-API-Key'] = key
        databases = request(session, args.api_url, 'GET', '/available_databases')
        if args.database not in databases:
            raise SystemExit('Requested database unavailable')
        metadata = databases[args.database]
        if 'database_metadata' in manifest and manifest['database_metadata'] != metadata:
            raise SystemExit('Database metadata changed: use a new output directory')
        manifest['database_metadata'] = metadata
        schema = request(session, args.api_url, 'GET', '/openapi.json')
        manifest['api_info'] = schema.get('info')
        save(args.out / 'api_schema.json', schema)
    save(manifest_path, manifest)
    save(args.out / 'queries.json', queries)
    save(args.out / 'run_status.json', {'phase': 'running', 'started_at': now(), 'pid': os.getpid()})
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(worker, args, metric, kind, queries, key): metric for metric, kind in METHODS.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
    save(args.out / 'run_status.json', {'phase': 'failed' if errors else 'complete', 'finished_at': now(), 'errors': errors})
    if errors:
        raise SystemExit('; '.join(errors))
    print(f"{now()} COMPLETE: {len(queries) * args.n} rows per CSV", flush=True)


if __name__ == '__main__':
    main()
