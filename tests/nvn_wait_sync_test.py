"""Condition-latched NVN recorder/submit regression, with no game or GPU.

Usage: python tests/nvn_wait_sync_test.py --cc /path/to/clang [--source-root BASELINE_CHECKOUT]
The subprocess timeout detects a deadlocked fixture; assertions use condition handshakes, never elapsed time.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with the project Clang executable')
    source = args.source_root.resolve()
    out = ROOT/'build'/'tests'/'nvn-wait-sync'
    out.mkdir(parents=True, exist_ok=True)
    exe = out/('nvn_wait_sync.exe' if os.name == 'nt' else 'nvn_wait_sync')
    # gfx_null.c is included by the fixture; compile from the selected source root for an exact baseline comparison.
    fixture = out/'nvn_wait_sync.c'
    fixture.write_text((ROOT/'tests/nvn_wait_sync.c').read_text().replace(
        '#include "../runtime/gfx_null.c"', '#include "' + (source/'runtime/gfx_null.c').as_posix() + '"'))
    subprocess.run([args.cc, '-std=gnu11', '-O1', '-Wall', '-Wextra', '-fno-strict-aliasing',
                    '-ffunction-sections', '-fdata-sections', '-I', str(source/'runtime'), str(fixture),
                    str(source/'runtime/nvn_api_cmd.c'), str(source/'runtime/nvn_api_core.c'),
                    '-Wl,--gc-sections', *([] if os.name == 'nt' else ['-pthread']), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)], timeout=20).returncode


if __name__ == '__main__':
    raise SystemExit(main())
