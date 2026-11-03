"""Exercise the production replay queue with condition latches; no game or GPU.

Usage: python tests/replay_drain_test.py --cc /path/to/clang [--source-root BASELINE_CHECKOUT]
The fixture timeout detects a deadlock, not success. Assertions do not depend on elapsed time or sleeping.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(text, declaration):
    start = text.index(declaration)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def declarations(text, first, last):
    start = text.index(first)
    return text[start:text.index(last, start)].rstrip()


def source(root):
    text = (root/'runtime/vk_backend.c').read_text(encoding='utf-8')
    # Keep the actual enum, item layout, capacity, shared state and statistics declarations, including whichever
    # counters the selected revision has. No counter updates or predicates are reconstructed by the fixture.
    parts = [
        declarations(text, 'enum { RI_SUBMIT', 'static const char *const dr_name'),
        declarations(text, 'static PlatMutex rl', '// Under rl:'),
        declarations(text, 'static uint64_t dr_calls', '// TOTK_REPLAY_CHECK:'),
    ]
    for declaration in ('static void replay_main(', 'static void wait_counted(', 'static int core_away_unlocked(',
                        'static void core_back_unlocked(', 'static uint64_t replay_enqueue(', 'static void replay_drain('):
        parts.append(function(text, declaration))
    return '\n\n'.join(parts) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with the project Clang executable')
    root = args.source_root.resolve()
    build = ROOT/'build/tests/replay-drain'
    build.mkdir(parents=True, exist_ok=True)
    (build/'replay_queue.inc').write_text(source(root), encoding='utf-8')
    exe = build/('replay_drain.exe' if os.name == 'nt' else 'replay_drain')
    subprocess.run([args.cc, '-std=gnu11', '-O1', '-Wall', '-Wextra', '-Werror', '-I', str(root/'runtime'),
                    '-I', str(build), str(ROOT/'tests/replay_drain.c'),
                    *([] if os.name == 'nt' else ['-pthread']), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)], timeout=20).returncode


if __name__ == '__main__':
    raise SystemExit(main())
