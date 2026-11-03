"""CPU regression for a deferred texture download overwriting a later mapped CPU write.

Requires Windows write-watch and Clang; does not link or execute Vulkan. The C fixture drives the actual
invalidate, guest-read, texture-writeback, SyncWait and FlushMappedRange code with a latched image-readback
adapter. A GPU fence becoming complete and a readback copying its result are separate events.

Usage: python tests/vk_guest_read_writeback_test.py --cc /path/to/clang [--revision COMMIT]
The original deferred implementation fails the race case. Its synchronous predecessor and both controls pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang'))
    parser.add_argument('--revision', help='read production source from this Git revision')
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('this fixture requires Windows write-watch')
    if not args.cc or args.runs < 1:
        parser.error('pass --cc and a positive --runs count')
    revision = subprocess.check_output(
        ['git', 'rev-parse', '--verify', '--end-of-options', args.revision + '^{commit}'],
        cwd=ROOT, text=True).strip() if args.revision else None
    out = ROOT/'build'/'tests'/'vk-guest-read-writeback'/(revision[:12] if revision else 'working')
    out.mkdir(parents=True, exist_ok=True)

    def source(name):
        path = 'runtime/' + name
        return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=ROOT, text=True) if revision else (ROOT/path).read_text()

    names = ('vk_backend.c', 'vk_texture.c', 'vk_memory.c', 'nvn_api_core.c', 'nvn_ext.c', 'nvn_tex.c', 'tex_deswizzle.c')
    sources = {name: source(name) for name in names}
    bodies, extracted = [], []

    def add(name, declaration):
        body = function(sources[name], declaration)
        bodies.append(body)
        extracted.append({'file': name, 'function': declaration, 'sha256': hashlib.sha256(body.encode()).hexdigest()})

    memory, backend, texture = (sources[name] for name in ('vk_memory.c', 'vk_backend.c', 'vk_texture.c'))
    generations = memory[memory.index('#define GEN_CHUNK_SHIFT'):memory.index('// Every write-watch query')]
    generations += '\nstatic uint32_t ww_ep = 1;\nstatic uint32_t reg_last[RT_ADDRESS_LIMIT >> REG_SHIFT];\n'
    region = texture[texture.index('typedef struct { unsigned level, layer0;'):texture.index('} Region;') + len('} Region;')]
    replay = backend[backend.index('enum { RI_SUBMIT'):backend.index('#define RQ_CAP')]
    for declaration in ('static uint32_t *gen_slot(', 'static void stamp_page(', 'static void stamp_range(',
                        'void vk_guest_written(', 'int vk_mem_changed('):
        add('vk_memory.c', declaration)
    add('nvn_api_core.c', 'void nvn_guest_written(')
    add('vk_backend.c', 'static void vk_be_guest_written(')
    for declaration in ('static int storage_rewritten(', 'static int region_of(', 'static int storage_texels(',
                        'static void write_back(', 'static int owned_lower(', 'static int owned_dirty(',
                        'int vk_texture_cpu_stale(', 'uint64_t vk_texture_dirty_fence(',
                        'void vk_texture_write_back(', 'int vk_texture_small_dirty('):
        add('vk_texture.c', declaration)
    add('vk_backend.c', 'static void vk_be_cpu_synced(')
    add('nvn_api_core.c', 'static uint64_t buf_address(')
    for declaration in ('NVN_API(nvnBufferFlushMappedRange)', 'NVN_API(nvnSyncWait)'):
        add('nvn_api_core.c', declaration)
    if 'static int read_nowait(' in backend:
        bodies.append('#define NOWAIT_MAX 4096\n#define NOWAIT_SEEN 8\n' + re.search(
            r'typedef struct \{ uint64_t addr, size, fence; \} NowaitSeen;', backend).group())
        add('vk_backend.c', 'static int nowait_covered(')
        add('vk_backend.c', 'static int read_nowait(')
    add('vk_backend.c', 'static void vk_be_guest_read(')
    add('nvn_ext.c', 'void nvn_pool_invalidate(')
    add('nvn_ext.c', 'NVN_API(nvnMemoryPoolInvalidateMappedRange)')

    fixture = (ROOT/'tests'/'vk_guest_read_writeback.c').read_text()
    for marker, value in (('GENERATION_GLOBALS', generations), ('REGION_TYPE', region),
                          ('REPLAY_TYPES', replay), ('PRODUCTION_FUNCTIONS', '\n\n'.join(bodies))):
        fixture = fixture.replace('/* ' + marker + ' */', value)
    (out/'fixture.c').write_text(fixture)
    (out/'nvn_tex.c').write_text(sources['nvn_tex.c'])
    deswizzle = sources['tex_deswizzle.c']
    if deswizzle.count('#include "tex.h"') != 1:
        raise ValueError('deswizzle include adapter no longer matches')
    # Include adapter only; the real layout and copy functions are compiled without body edits.
    (out/'tex_deswizzle.c').write_text(deswizzle.replace('#include "tex.h"', '#include "nvn_tex.h"'))
    results = []
    for optimization in ('-O0', '-O2'):
        exe = out/('vk_guest_read_writeback' + optimization[1:] + '.exe')
        command = [args.cc, '-std=gnu11', optimization, '-Wall', '-Wextra', '-Werror',
                   '-fno-strict-aliasing', '-I', str(ROOT/'runtime'), str(out/'fixture.c'),
                   str(out/'nvn_tex.c'), str(out/'tex_deswizzle.c'), '-o', str(exe)]
        subprocess.run(command, check=True)
        for scenario in ('race', 'before', 'stable'):
            rows = []
            for _ in range(args.runs):
                result = subprocess.run([str(exe), scenario], capture_output=True, text=True, timeout=15)
                rows.append({'exit': result.returncode, 'output': result.stdout + result.stderr})
            failures = sum(row['exit'] != 0 for row in rows)
            print(f'{optimization} {scenario}: {len(rows) - failures}/{len(rows)} pass; {rows[0]["output"].strip()}')
            results.append({'optimization': optimization, 'scenario': scenario, 'runs': rows})
    (out/'results.json').write_text(json.dumps({
        'revision': revision or 'working tree',
        'source_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
        'extracted_functions': extracted,
        'fixture_sha256': hashlib.sha256(fixture.encode()).hexdigest(),
        'scope': 'Actual production CPU paths; image readback and a single-item replay scheduler are explicit adapters; no Vulkan.',
        'results': results,
    }, indent=2) + '\n')
    return int(any(row['exit'] for result in results for row in result['runs']))


if __name__ == '__main__':
    raise SystemExit(main())
