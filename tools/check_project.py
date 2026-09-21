#!/usr/bin/env python3
"""Dataset-free checks of release structure, source syntax, and saved evidence."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(folder, optional=False):
    count = 0
    for line in (folder / 'MANIFEST.sha256').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = folder / relative
        assert path.resolve().is_relative_to(folder.resolve()), relative
        if optional and not path.exists():
            continue
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Changed evidence: {path}'
        count += 1
    return count


def main():
    for name in ['README.md', 'LICENSE', 'CITATION.cff', 'THIRD_PARTY.md',
                 'protocol/metadata.csv', 'results/README.md',
                 'environments/sources.json']:
        assert (ROOT / name).is_file(), name
    for name in ['archive', 'job_logs', 'rebuttal', 'Reviews.txt', 'tasks.md',
                 'pilot_outputs', 'pilot_pipeline_steps', 'ACCV_Paper']:
        assert not (ROOT / name).exists(), f'Unexpected historical item: {name}'
    for folder in ['pipeline', 'gait_integration', 'pbp_integration', 'analysis', 'tools', 'paper']:
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    evidence = verify_manifest(ROOT / 'results')
    protocol = verify_manifest(ROOT / 'protocol')
    artifacts = verify_manifest(ROOT / 'artifacts', optional=True)
    sources = json.loads((ROOT / 'environments/sources.json').read_text())
    for source in sources:
        assert re.fullmatch(r'[0-9a-f]{40}', source['revision'])
        for patch in source['patches']:
            assert (ROOT / 'environments' / patch).is_file()
    # Detect developer-specific paths without matching the checker itself.
    private_roots = ['/' + 'SLURM/home/', '/' + 'raid_storage/', 'C:' + '\\Users\\']
    for folder in ['pipeline', 'analysis', 'tools', 'jobs', 'gait_integration',
                   'pbp_integration', 'docs', 'environments', 'results', 'protocol']:
        for path in (ROOT / folder).rglob('*'):
            if path.suffix not in {'.py', '.sh', '.sbatch', '.md', '.csv', '.json', '.txt', '.patch'}:
                continue
            text = path.read_text()
            assert not any(token in text for token in private_roots), f'Private machine path: {path}'
    for path in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in link or link.startswith('#'):
                continue
            assert (path.parent / link.split('#')[0]).exists(), f'Broken link in {path}: {link}'
    with (ROOT / 'results/runtime.csv').open(newline='') as f:
        timings = list(csv.DictReader(f))
    assert len(timings) == 4 and sum(int(r['targets']) for r in timings) == 3600
    assert abs(sum(float(r['elapsed_minutes']) for r in timings) - 650.5) < 1e-8
    print(f'PASS: release structure, Python syntax, {protocol} protocol files, '
          f'{evidence} result files, {artifacts} available local artifacts, and portable paths.')


if __name__ == '__main__':
    main()
