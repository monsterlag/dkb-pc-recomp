"""Run original masked-stencil dispatcher regressions without a Vulkan device.

Use --source-root on a prior checkout to confirm the skipped-mask failure. The fixture extracts the production
clear dispatcher unchanged; Vulkan calls and texture/pass bookkeeping are adapters. Use --probe-vkclear for actual
production helper shaders, fixed-function pipeline and GPU readback with validation.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', default=shutil.which('clang') or 'clang')
    ap.add_argument('--deps', type=Path, required=True, help='runtime deps directory containing include/vulkan and include/vma')
    ap.add_argument('--source-root', type=Path, default=ROOT)
    args = ap.parse_args()
    prod = (args.source_root / 'runtime/vk_exec.c').read_text()
    code = '\n\n'.join(function(prod, f'static {decl}(') for decl in
                       ('int whole_target', 'void clear_attachment', 'void clear_ds'))
    fixture = (ROOT / 'tests/stencil_clear_test.c').read_text().replace(
        '/* production functions are inserted here by stencil_clear_test.py */', code)
    with tempfile.TemporaryDirectory(prefix='dkb-stencil-') as tmp:
        src = Path(tmp) / 'test.c'
        src.write_text(fixture)
        for opt in ('-O0', '-O2'):
            exe = Path(tmp) / ('stencil' + opt[2:] + '.exe')
            subprocess.run([args.cc, '-std=gnu11', opt, '-I', str(ROOT / 'runtime'),
                            '-I', str(args.deps / 'include'), '-I', str(args.deps / 'include/vma'),
                            str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
