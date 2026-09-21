#!/usr/bin/env python3
"""Fetch pinned external code into the user cache; never overwrite source trees."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Show sources without downloading.')
    parser.add_argument('--only', action='append', help='Source basename; repeat to select several.')
    cache = Path(os.environ.get('ACCV_CACHE', Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'accv'))
    parser.add_argument('--destination', type=Path, default=Path(os.environ.get('REPOS_DIR', cache / 'repos')))
    args = parser.parse_args()
    sources = json.loads((ROOT / 'environments/sources.json').read_text())
    known = {Path(s['path']).name for s in sources}
    if args.only and not set(args.only) <= known:
        parser.error('Unknown source; use --list to see available names.')
    for source in sources:
        if args.only and Path(source['path']).name not in args.only:
            continue
        target = args.destination.expanduser().resolve() / source['path']
        print(f"{source['path']} @ {source['revision']} -> {target}", flush=True)
        if args.list:
            continue
        if target.exists():
            raise SystemExit(f'Refusing to change existing source tree: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'init', str(target)], check=True)
        def git(*command):
            subprocess.run(['git', '-C', str(target), *command], check=True)
        git('remote', 'add', 'origin', source['url'])
        git('fetch', '--depth', '1', 'origin', source['revision'])
        git('-c', 'advice.detachedHead=false', 'checkout', '--detach', 'FETCH_HEAD')
        for relative in source['patches']:
            patch = str(ROOT / 'environments' / relative)
            git('apply', '--check', patch)
            git('apply', patch)


if __name__ == '__main__':
    main()
