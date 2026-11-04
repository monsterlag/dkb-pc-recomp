"""Original CPU regression for actual audio routing/render; no SDL device/library.

Actual wire structs, Mixer/voice state, routing parser, decode/render, effect
dispatcher and splitter filter are used. Output/gain/metering are providers;
unexpected DSP/circular-sink/logger calls abort. Sources contain no game data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    position = source.index('{', start)
    depth = 1
    end = position + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def generate(source):
    headers = '''#include <stdint.h>
#include <stdatomic.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "audren.h"
#include "audio_effects.h"
#include "audio_reverb.h"
#include "host.h"
typedef struct SDL_AudioStream SDL_AudioStream; // opaque type only; SDL is not linked
'''
    types = source[source.index('#define MAX_FRAME'):source.index('// TOTK_VOLUME:')]
    routing = source[source.index('static void splitters_update('):source.index('/* Effect parameters arrive')]
    decoding = source[source.index('typedef struct { ServerVoice *v;'):source.index('/* ---------- mixing ---------- */')]
    mixing = source[source.index('static float *mixbuf('):source.index('/* Guest custom effects exchange')]
    production = '\n'.join([
        function((ROOT / 'runtime/audren.c').read_text(), 'int audio_buffer_empty('),
        function(source, 'static void report_once('), routing, decoding, mixing,
        function(source, 'static void mix_effects('),
        function(source, 'static const float *splitter_filter('),
        function(source, 'void audio_mixer_render('),
    ])
    fixture = (ROOT / 'tests/audio_splitter_order.c').read_text()
    if fixture.count('/* @PRODUCTION@ */') != 1:
        raise RuntimeError('unexpected fixture marker')
    return headers + types + fixture.replace('/* @PRODUCTION@ */', production)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--opt', choices=('0', '2'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mixer-source', type=Path, help='optional exact saved baseline audio_mixer.c')
    args = parser.parse_args()
    out = args.out.resolve()
    out.relative_to((ROOT / 'build').resolve())
    out.mkdir(parents=True, exist_ok=False)
    path = args.mixer_source or ROOT / 'runtime/audio_mixer.c'
    fixture = out / 'fixture.c'
    fixture.write_text(generate(path.read_text()))
    exe = out / 'fixture.exe'
    command = [args.cc, '-std=gnu11', '-O' + args.opt, '-Wall', '-Wextra', '-Werror',
               '-I', ROOT / 'runtime', fixture, '-o', exe]
    compiled = subprocess.run([str(x) for x in command], capture_output=True, text=True, timeout=60)
    (out / 'compile.log').write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
    (out / 'test.log').write_text(result.stdout + result.stderr)
    manifest = {'result': result.stdout.strip(), 'exit_code': result.returncode,
                'optimization': args.opt, 'mixer_source': str(path),
                'mixer_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
                'sources_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in ('runtime/audren.c', 'runtime/audren.h', 'runtime/audio_effects.h',
                                 'runtime/audio_reverb.h', 'tests/audio_splitter_order.c',
                                 'tests/audio_splitter_order_test.py')},
                'providers': ['owned original PCM-float server voices; no guest capture',
                              'actual full REV15 splitter/mix input parser and render/decode functions',
                              'output capture and unit host/device gain; metering has no routing effect',
                              'unexpected DSP/circular-sink/logger abort; fatal cycle captured with setjmp',
                              'opaque SDL pointer type only; no SDL imports/device/audio output'],
                'limits': ['no audible/game verification or issue63 route attribution',
                           'no numerical DSP/filter or resampling quality validation',
                           'out-of-range splitter case preserves current host fallback policy, not SDK legality']}
    (out / 'handoff.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
