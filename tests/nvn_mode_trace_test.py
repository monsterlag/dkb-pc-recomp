"""Windows CPU test of actual mode trace, NVN wrappers and recorder/prepare hooks.

Platform locking/once/thread-ID and cfg/logger are explicit fixture providers.
Renderer and guest callbacks abort. No modules, driver or Vulkan loader is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cc', required=True)
    args = ap.parse_args()
    out = ROOT / 'build/checks/nvn2-mode-trace'
    out.mkdir(parents=True, exist_ok=True)
    sources = ['tests/nvn_mode_trace.c', 'runtime/nvn_mode_trace.c', 'runtime/nvn_api_cmd.c', 'runtime/nvn_api_nvn2.c']
    fixture = out / 'fixture.c'
    fixture.write_text((ROOT / sources[0]).read_text().replace('@PROF_WRITE@',
        function((ROOT / 'runtime/perf.c').read_text(), 'void prof_write(void)')))
    rows = []
    for opt in ('0', '2'):
        exe = out / f'trace-o{opt}.exe'
        command = [args.cc, '-std=gnu11', '-O'+opt, '-Wall', '-Wextra', '-Werror',
                   '-Wno-missing-field-initializers',  # inherited copy_prefix initializer in the linked recorder
                   '-ffunction-sections', '-fdata-sections', '-I', ROOT / 'runtime',
                   fixture, *[ROOT / name for name in sources[1:]], '-Wl,--gc-sections', '-o', exe]
        compiled = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=60)
        (out / f'compile-o{opt}.log').write_text(compiled.stdout + compiled.stderr)
        if compiled.returncode:
            raise RuntimeError(compiled.stderr)
        for mode in ('off', 'on', 'race', 'window', 'empty', 'open', 'max', 'bad-start', 'bad-duration'):
            result = subprocess.run([str(exe), mode], capture_output=True, text=True, timeout=30)
            (out / f'{mode}-o{opt}.log').write_text(result.stdout + result.stderr)
            rows.append({'optimization': opt, 'mode': mode, 'exit': result.returncode, 'output': result.stdout.strip()})
            print(result.stdout, end='')
            if result.returncode:
                print(result.stderr, end='')
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    hashes['runtime/perf.c'] = hashlib.sha256((ROOT / 'runtime/perf.c').read_bytes()).hexdigest()
    (out / 'handoff.json').write_text(json.dumps({'results': rows, 'source_sha256': hashes,
        'providers': 'Win32 SRW locks/threads/events; atomic once adapter; fake restart cfg; scalar logger parser; guest/backend abort',
        'race_proof': 'writer holds real helper lock inside thread-ID provider; flush attempts that lock before writer is released'}, indent=2)+'\n')
    return int(any(row['exit'] for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
