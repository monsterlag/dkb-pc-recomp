"""Original storage-candidate compute readback; preparation is CPU-only, GPU execution requires --run.

Uses storage_candidates_fixture's original instruction encoders. Fake guest addresses are uniform integers
mapped to owned input/output buffers. Every execution requires robustBufferAccess. Baseline wrong offsets stay
inside the legacy 64-KiB dummy; guarded invalid candidates bind one byte, giving uint ArrayLength zero.
Both capability modes use the same expected data assertion; baseline RED results are never treated as passes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess

import storage_candidates_fixture as fixture

ROOT = Path(__file__).resolve().parents[1]
MAGIC = 0x53434750
ALIGN = 256
GUARD = 256
POOL_BYTES = 4096
DUMMY_BYTES = 65536
UBO_BYTES = 65536
VALUE = 0x1234ABCD
OUTPUT_TOKEN = 0x500000000
NAMES = ('literal', 'inline', 'opposite', 'only_c1', 'missing_c1', 'short_c1')
WIDE_CASES = {
    'wide64_literal': (64, False, 0), 'wide64_inline': (64, True, 0),
    'wide128_literal': (128, False, 0), 'wide128_inline': (128, True, 0),
    'wide64_inline_imm32': (64, True, 32), 'wide128_inline_imm32': (128, True, 32),
}
ALL_NAMES = (*NAMES, *WIDE_CASES, 'exact_inline')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spirv_requirements(path):
    data = path.read_bytes()
    require(len(data) >= 20 and len(data) % 4 == 0, 'invalid SPIR-V size')
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    require(words[0] == 0x07230203, 'invalid SPIR-V magic')
    at, caps, entries, bindings, sets, names, variables, pointers = 5, set(), [], {}, {}, {}, {}, {}
    while at < len(words):
        n, op = words[at] >> 16, words[at] & 65535
        require(n and at + n <= len(words), 'invalid SPIR-V instruction length')
        a = words[at + 1:at + n]
        if op == 17:
            caps.add(a[0])
        elif op == 5:
            names[a[0]] = struct.pack('<' + 'I' * (len(a) - 1), *a[1:]).split(b'\0')[0].decode()
        elif op == 59:
            variables[a[1]] = (a[0], a[2])
        elif op == 32:
            pointers[a[0]] = (a[1], a[2])
        elif op == 15:
            entries.append(a)
        elif op == 71 and a[1] == 33:
            bindings[a[0]] = a[2]
        elif op == 71 and a[1] == 34:
            sets[a[0]] = a[2]
        at += n
    require(len(entries) == 1 and entries[0][0] == 5, 'expected one compute entry point')
    require(all(sets.get(ident) == 0 for ident in bindings), 'fixture supports descriptor set zero only')
    require(caps <= {1, 10, 11, 25, 46, 47, 50, 61, 62, 64, 65}, 'unexpected shader capability')
    features = int(10 in caps) | (int(11 in caps) << 1) | (int(25 in caps) << 2)
    subgroup = int(62 in caps) | (int(64 in caps) << 1) | (int(65 in caps) << 2)
    support = [ident for ident, binding in bindings.items() if binding == 0]
    require(len(support) == 1 and names.get(support[0]) == 'support_buffer' and
            variables[support[0]][1] == 2 and pointers[variables[support[0]][0]][0] == 2,
            'expected the translator\'s reserved support_buffer uniform binding0')
    return features, subgroup, sorted(bindings.values()), sorted(caps)


def storage(size, fill):
    return bytearray(bytes([0xA6]) * GUARD + bytes([fill]) * size + bytes([0x6A]) * GUARD)


def case_data(name, carry):
    wide = WIDE_CASES.get(name)
    memory_offset = wide[2] if wide else 0
    exact = name == 'exact_inline'
    require(not exact or not carry, 'exact low-bits4 case supports the bounded normal pointer only')
    pointer = 0x200000004 if exact else (0x2FFFFFFC0 if carry and memory_offset else
                                        0x2FFFFFFF0 if carry else 0x200000000)
    address_alignment = 0 if exact else ALIGN
    pool_base = pointer if exact else pointer & -ALIGN
    c0, c1 = bytearray(UBO_BYTES), bytearray(UBO_BYTES)
    opposite = name in ('opposite', 'only_c1')
    struct.pack_into('<Q', c0, 16, 32 if opposite else pointer)
    struct.pack_into('<Q', c0, 32, OUTPUT_TOKEN)
    struct.pack_into('<Q', c1, 0, pointer if opposite else 32)
    extra = 0 if name == 'only_c1' else 32
    values = tuple(fixture.WIDE_VALUES[:wide[0] // 32]) if wide else (VALUE,)
    source_offset = pointer - pool_base + extra + memory_offset
    require(source_offset + 4 * len(values) <= POOL_BYTES, 'source leaves owned input pool')
    require(not wide or (pointer + extra + memory_offset) % (wide[0] // 8) == 0,
            'wide fixture must be naturally aligned')
    buffers = [storage(POOL_BYTES, 0xB7), storage(POOL_BYTES, 0xCD), storage(DUMMY_BYTES, 0),
               storage(UBO_BYTES, 0), storage(UBO_BYTES, 0)]
    struct.pack_into('<' + 'I' * len(values), buffers[0], GUARD + source_offset, *values)
    initial = (0xA5A5A5A5,) * len(values) if wide else (0xDEADBEEF,)
    require(all(a != b for a, b in zip(initial, values)), 'output sentinel must differ from every expected word')
    struct.pack_into('<' + 'I' * len(values), buffers[1], GUARD + memory_offset, *initial)
    buffers[3][GUARD:GUARD + UBO_BYTES] = c0
    buffers[4][GUARD:GUARD + UBO_BYTES] = c1
    require(not carry or wide or name == 'only_c1' or source_offset == 272, 'scalar carry fixture must use logical offset272')
    # The old path truncates its incorrect address difference to16 or32; both are inside the zero dummy.
    if name in ('inline', 'opposite', 'missing_c1', 'short_c1', 'exact_inline') or wide and wide[1]:
        # Exact-mode's legacy offset is the c0 low word; explicit-alignment legacy uses the truncated full address.
        old_offset = (pointer if exact else pointer + extra + memory_offset) & 0xFFFFFFFF
        require(old_offset + 4 * len(values) <= DUMMY_BYTES, 'baseline dummy access would be unbounded')
    return {'pointer': pointer, 'pool_base': pool_base, 'source_offset': source_offset,
            'c0': c0, 'c1': c1, 'buffers': buffers, 'expected_values': values,
            'output_offset': memory_offset, 'address_alignment': address_alignment,
            'access_bits': wide[0] if wide else 32, 'memory_offset': memory_offset}


def packet(path, reflection, data, requirements, capability):
    features, subgroup, spv_bindings, caps = requirements
    require(reflection.get('storage_candidates_version', 0) == capability, 'candidate capability echo mismatch')
    require(not reflection['textures'] and not reflection['images'], 'unexpected texture/image resource')
    # The translator declares its reserved support UBO separately from cbuf reflection. Supply valid zero storage.
    bindings = [(0, 0, 2, GUARD, DUMMY_BYTES)]
    description = [{'binding': 0, 'type': 'support_uniform', 'buffer': 'dummy', 'offset': GUARD, 'range': DUMMY_BYTES}]
    for r in reflection['cbufs']:
        bank = r['slot']
        require(bank in (0, 1), 'unexpected uniform bank')
        bindings.append((r['binding'], 0, 3 + bank, GUARD, UBO_BYTES))
        description.append({'binding': r['binding'], 'type': 'uniform', 'bank': bank, 'range': UBO_BYTES})
    identities = {}
    for r in reflection['sbufs']:
        bank, off, flags = r['cb_slot'], r['cb_offset'] * 4, r['flags']
        require(bank in (0, 1) and not (flags & ~0x81), 'unexpected storage resource metadata')
        require(0 <= off <= UBO_BYTES - 8, 'storage pointer pair leaves owned uniform bank')
        guarded = bool(flags & 0x80)
        require(capability == 1 or not guarded, 'legacy reflection unexpectedly requests guarded fallback')
        value = struct.unpack_from('<Q', data['c0'] if bank == 0 else data['c1'], off)[0]
        aligned = value & -data['address_alignment'] if data['address_alignment'] else value
        buf, offset, size, backed = 2, GUARD, 1 if guarded else DUMMY_BYTES, False
        for index, base in ((0, data['pool_base']), (1, OUTPUT_TOKEN)):
            if base <= aligned < base + POOL_BYTES:
                buf, offset, size, backed = index, GUARD + aligned - base, POOL_BYTES - (aligned - base), True
                break
        identity = (bank, r['cb_offset'], guarded)
        require(identity not in identities, 'duplicate storage identity')
        identities[identity] = r['binding']
        bindings.append((r['binding'], 1, buf, offset, size))
        description.append({'binding': r['binding'], 'type': 'storage', 'bank': bank, 'word_offset': r['cb_offset'],
                            'flags': flags, 'guarded': guarded, 'guest_address': hex(value), 'aligned_guest_base': hex(aligned),
                            'buffer': ('input', 'output', 'dummy')[buf], 'offset': offset, 'range': size, 'backed': backed})
    require(0 < len(bindings) <= 32, 'unsupported descriptor count')
    require(len({b[0] for b in bindings}) == len(bindings), 'descriptor binding collision')
    require(sorted(b[0] for b in bindings) == spv_bindings, 'reflection and SPIR-V bindings differ')
    require(any(b[1] == 1 and b[2] == 1 for b in bindings), 'fixture has no output storage binding')
    sizes = [len(b) for b in data['buffers']]
    header = struct.pack('<13I', MAGIC, 1, len(bindings), 5, features, subgroup, data['address_alignment'], 0, *sizes)
    path.write_bytes(header + b''.join(struct.pack('<5I', *b) for b in bindings) + b''.join(data['buffers']))
    return {'bindings': description, 'buffer_sizes': sizes, 'capabilities': caps,
            'features': features, 'subgroups': subgroup, 'packet_sha256': digest(path)}


def compare_readback(path, buffers, expected_values=(VALUE,), output_offset=0):
    raw = path.read_bytes()
    require(len(raw) == sum(map(len, buffers)), 'incorrect readback length')
    at, result = 0, {}
    for index, name in enumerate(('input', 'output', 'dummy', 'c0', 'c1')):
        expected = bytearray(buffers[index])
        got = raw[at:at + len(expected)]
        at += len(expected)
        if name == 'output':
            require(expected_values and output_offset + 4 * len(expected_values) <= POOL_BYTES,
                    'output expectation leaves the owned pool')
            actual = struct.unpack_from('<' + 'I' * len(expected_values), got, GUARD + output_offset)
            result['actual_output_words'] = list(actual)
            result['expected_output_words'] = list(expected_values)
            result['output_word_matches'] = [a == b for a, b in zip(actual, expected_values)]
            result['actual_output'] = actual[0]  # retain scalar report fields
            result['expected_output'] = expected_values[0]
            result['output_matches'] = all(result['output_word_matches'])
            lo, hi = GUARD + output_offset, GUARD + output_offset + 4 * len(expected_values)
            expected[lo:hi] = got[lo:hi]
        result[name + '_canaries_unchanged'] = got == expected
    result['semantic_pass'] = result['output_matches'] and all(v for k, v in result.items() if k.endswith('_unchanged'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translator', required=True, type=Path)
    parser.add_argument('--spirv-val', required=True, type=Path)
    parser.add_argument('--cc', required=True)
    parser.add_argument('--vulkan-include', required=True, type=Path)
    parser.add_argument('--vulkan-library', required=True, type=Path)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/tests/storage-candidates-gpu')
    parser.add_argument('--capabilities', default='1', help='0, 1, or 0,1; capability0 retains baseline RED assertions')
    parser.add_argument('--cases', default=','.join(ALL_NAMES))
    parser.add_argument('--pointers', default='normal,carry')
    parser.add_argument('--device-name', default='')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--run', action='store_true', help='Explicitly execute Vulkan/GPU work after preparation')
    args = parser.parse_args()
    capabilities = [int(x) for x in args.capabilities.split(',')]
    names, pointers = args.cases.split(','), args.pointers.split(',')
    require(capabilities and len(set(capabilities)) == len(capabilities) and set(capabilities) <= {0, 1}, 'invalid capabilities')
    require(names and len(set(names)) == len(names) and set(names) <= set(ALL_NAMES), 'invalid cases')
    require(pointers and len(set(pointers)) == len(pointers) and set(pointers) <= {'normal', 'carry'}, 'invalid pointers')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    runner = out / ('storage-candidates-gpu.exe' if os.name == 'nt' else 'storage-candidates-gpu')
    cc = shutil.which(args.cc) or args.cc
    compile_command = [str(cc), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-I', str(args.vulkan_include.resolve()),
                       str(ROOT / 'tests/storage_candidates_gpu.c'), str(args.vulkan_library.resolve()), '-o', str(runner)]
    build = subprocess.run(compile_command, capture_output=True, text=True, timeout=60)
    (out / 'compile.log').write_text(build.stdout + build.stderr)
    require(build.returncode == 0, build.stdout + build.stderr)
    requests, cases, omitted = [], [], []
    for capability in capabilities:
        for name in names:
            for pointer in pointers:
                if name == 'exact_inline' and pointer == 'carry':
                    omitted.append({'capability': capability, 'case': name, 'pointer_case': pointer,
                                    'reason': 'exact low-bits4 fixture uses only its bounded normal token'})
                    continue  # this new native/exact fixture deliberately uses only its bounded low-bits4 pointer
                key = f'v{capability}_{name}_{pointer}'
                folder = out / key
                folder.mkdir()
                data = case_data(name, pointer == 'carry')
                code = folder / 'original.bin'
                wide = WIDE_CASES.get(name)
                code.write_bytes(fixture.wide_shader_bytes(*wide) if wide else
                                 fixture.c1_pointer_bytes(name == 'opposite') if name in ('opposite', 'only_c1')
                                 else fixture.shader_bytes(name != 'literal'))
                inline = struct.pack('<Q', data['pointer'] if name in ('opposite', 'only_c1') else 32)
                job = dict(key=key, stage='cs', code=str(code), debug=False, c1=(inline + bytes(248)).hex(),
                           local=[1, 1, 1], shared=0, c0=[0, 0, 0, 0], c0_version=2,
                           c0_roots=[8, 0, 0, 0, 0, 0, 0, 0], sbuf_align=ALIGN, storage_candidates_version=capability)
                if not data['address_alignment']:
                    del job['sbuf_align']
                if name == 'missing_c1':
                    del job['c1']
                elif name == 'short_c1':
                    job['c1'] = inline[:7].hex()
                requests.append({'job': job, 'out': str(folder)})
                cases.append((key, capability, name, pointer, folder, data))
    require(cases, 'selected cases/pointers produce no supported fixtures')
    env = {k: v for k, v in os.environ.items() if k.upper() in {'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP'}}
    translation = subprocess.run([str(args.translator.resolve()), '--serve'], env=env,
        input=''.join(json.dumps(r) + '\n' for r in requests), capture_output=True, text=True, timeout=120)
    (out / 'translation.log').write_text(translation.stdout + translation.stderr)
    require(translation.returncode == 0 and translation.stdout.splitlines() == ['ok ' + c[0] for c in cases],
            translation.stdout + translation.stderr)
    manifest = {'gpu_execution_requested': args.run, 'gpu_processes_completed': 0,
                'translator_sha256': digest(args.translator), 'runner_sha256': digest(runner),
                'fixture_encoder_sha256': digest(ROOT / 'tests/storage_candidates_fixture.py'),
                'compile_command': compile_command, 'cases': [], 'omitted_combinations': omitted, 'limitations': [
                    'Original naturally aligned non-atomic B32/B64/B128 load/store fixtures only; no captured code, atomics or alias-coherence claim.',
                    'Wide vectors test defined whole-span accesses. Last-natural-start wide GPU variants are excluded because incorrect baseline offsets could leave the dummy.',
                    'The exact/no-sbuf_align case has only a normal low-bits4 token; its owned physical descriptor remains device-aligned without rounding the fake guest base.',
                    'Shader addresses are fake uniform tokens. Descriptor mapping is owned test code, not production ssbo_info.',
                    'Missing/short compile metadata uses a complete safe runtime UBO; it is not an out-of-bounds GPU test.',
                    'Every run requires robustBufferAccess. Validation availability/request/enabling are reported separately.',
                    'Input/dummy and output bytes outside the result, including descriptor-external guard bands, must remain unchanged.']}
    for key, capability, name, pointer, folder, data in cases:
        spv = folder / (key + '.spv')
        validation = subprocess.run([str(args.spirv_val.resolve()), '--target-env', 'vulkan1.3', str(spv)],
                                    capture_output=True, text=True, timeout=30)
        (folder / 'spirv-validation.log').write_text(validation.stdout + validation.stderr)
        require(validation.returncode == 0, validation.stdout + validation.stderr)
        reflection = json.loads((folder / (key + '.res.json')).read_text())
        packet_path = folder / 'bindings.bin'
        row = {'key': key, 'capability': capability, 'case': name, 'pointer_case': pointer,
               'input_pointer': hex(data['pointer']), 'input_pool_base': hex(data['pool_base']),
               'source_logical_offset': data['source_offset'], 'output_logical_offset': data['output_offset'],
               'access_bits': data['access_bits'], 'memory_offset': data['memory_offset'],
               'host_address_alignment': data['address_alignment'], 'spv_sha256': digest(spv),
               **packet(packet_path, reflection, data, spirv_requirements(spv), capability)}
        if name in ('missing_c1', 'short_c1'):
            cb1 = [r for r in reflection['cbufs'] if r['slot'] == 1]
            require(len(cb1) == 1 and fixture.loads_uniform_word_zero(spv.read_bytes(), cb1[0]['binding']),
                    'unknown c1 must remain an actual runtime uniform load')
        if args.run:
            readback = folder / 'readback.bin'
            command = [str(runner), str(spv), str(packet_path), str(readback), args.device_name, str(int(args.validation))]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
                row['pid'] = process.pid
                (folder / 'launch.json').write_text(json.dumps({'pid': process.pid, 'command': command}, indent=2) + '\n')
                try:
                    stdout, stderr = process.communicate(timeout=90)
                except subprocess.TimeoutExpired:
                    process.kill()  # this helper's child only; stop the matrix after recording its timeout
                    stdout, stderr = process.communicate(timeout=15)
                    row['timeout_seconds'] = 90
            (folder / 'stdout.log').write_text(stdout)
            (folder / 'stderr.log').write_text(stderr)
            row['runner_exit'] = process.returncode
            reports = [line for line in stdout.splitlines() if line.startswith('{')]
            row['device_report'] = json.loads(reports[-1]) if reports else None
            row['readback'] = compare_readback(readback, data['buffers'], data['expected_values'], data['output_offset']) if readback.exists() else {'semantic_pass': False}
            report = row['device_report'] or {}
            row['passed'] = (process.returncode == 0 and row['readback']['semantic_pass'] and
                             report.get('robust_buffer_access_enabled') is True and
                             report.get('validation_requested') is args.validation and
                             report.get('validation_errors') == 0)
            manifest['gpu_processes_completed'] += 1
            (folder / 'result.json').write_text(json.dumps(row, indent=2) + '\n')
        manifest['cases'].append(row)
        # Preserve completed evidence if a later process fails or times out.
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        require('timeout_seconds' not in row, 'owned GPU helper timed out; matrix stopped')
        if args.run:
            require(row['runner_exit'] == 0, 'GPU execution or validation failed; completed evidence retained and matrix stopped')
    failures = sum(not r['passed'] for r in manifest['cases']) if args.run else None
    print(json.dumps({'prepared': len(cases), 'gpu_executed': args.run, 'failures': failures, 'manifest': str(out / 'manifest.json')}))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
