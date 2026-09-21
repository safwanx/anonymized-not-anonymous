#!/usr/bin/env python3
"""Submit a repository SLURM template with portable root and log paths."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--dependency')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    job = args.job if args.job.is_absolute() else ROOT / args.job
    job = job.resolve()
    if not job.is_relative_to(ROOT / 'jobs') or not job.is_file() or job.suffix != '.sbatch':
        parser.error('Select a .sbatch template from this repository\'s jobs directory.')
    output = Path(os.environ.get('PILOT_OUTPUT_ROOT', ROOT / 'outputs')).expanduser().resolve()
    logs = output / 'logs'
    command = ['sbatch', '--chdir', str(ROOT), '--partition', args.partition,
               '--output', str(logs / '%x-%A_%a.out'), '--error', str(logs / '%x-%A_%a.err')]
    if args.dependency:
        command += ['--dependency', args.dependency]
    command.append(str(job))
    print(shlex.join(command))
    if not args.dry_run:
        logs.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, ACCV_ROOT=str(ROOT), PILOT_OUTPUT_ROOT=str(output))
        subprocess.run(command, env=environment, check=True)


if __name__ == '__main__':
    main()
