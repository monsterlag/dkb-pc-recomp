"""Compile/run synthetic NVN 2 compute c0 tests; --baseline REV checks an earlier production packer.

Usage: python tests/nvn2_compute_c0_test.py [--cc clang] [--baseline REV]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(source, start):
    begin = source.index(start)
    at = source.index('{', begin)
    depth = 1
    end = at + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    parser.add_argument('--baseline', metavar='REV', help='run the production packer at an earlier Git revision')
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    out = ROOT / 'build' / 'test' / 'nvn2-compute-c0'
    out.mkdir(parents=True, exist_ok=True)
    exe = out / ('nvn2_compute_c0.exe' if os.name == 'nt' else 'nvn2_compute_c0')
    extra = []
    if args.baseline:
        # Keep the original production function intact; only replace unrelated global/replay context.
        source = subprocess.check_output(['git', 'show', f'{args.baseline}:runtime/vk_exec_draw.c'], cwd=ROOT, text=True)
        shim = out / 'baseline.c'
        shim.write_text('''#include "nvn2_c0.h"
#define VK_C0_BYTES NVN2_C0_BYTES
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_CS 5
static const uint8_t *c0_lay;
static uint32_t c0_imax[8][2];
''' + function(source, 'static void c0_table(') + '\n' + function(source, 'void vk_c0_fill(') + '''
int baseline_pack(uint8_t *bank, size_t capacity, const Nvn2C0Layout *l,
                  const uint32_t roots[8][64], const p_bind_buf ssbo[16], const p_bind_buf ubo[16]) {
    (void)roots;
    if (!nvn2_c0_valid(l, capacity)) return 0;
    c0_lay = (const uint8_t *)l;
    vk_c0_fill(bank, NVN_STAGE_CS, ssbo, ubo, (const uint32_t[3]){0}, 0);
    return 1;
}
''')
        extra = ['-DNVN2_C0_BASELINE', str(shim)]
    subprocess.run([args.cc, '-O2', '-Wall', '-Wextra', '-I', str(ROOT / 'runtime'),
                    str(ROOT / 'tests' / 'nvn2_compute_c0.c'), *extra, '-o', str(exe)], check=True)
    return subprocess.run([str(exe)]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
