"""Read back the owned translator fixture's math on a headless Vulkan compute queue.

Preparation translates only float_minmax_test.shader(), validates a deliberately narrow fragment-to-compute
adapter, and compiles the runner. GPU execution requires --run. Strict special-value comparisons describe
the observed Maxwell target contract, not Vulkan conformance under an implicit floating-point environment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import float_minmax_test as fixture

ROOT = Path(__file__).resolve().parents[1]
FP_MODES = {4459: 'denorm_preserve', 4460: 'denorm_flush', 4461: 'signed_zero_inf_nan_preserve',
            4462: 'round_rte', 4463: 'round_rtz'}
UNUSED_CAPABILITIES = {10, 25, 46, 47, 50, 62, 64, 65, 5378}
ALLOWED_OPS = {5, 10, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 28, 30, 32, 33, 43,
               54, 56, 59, 61, 62, 65, 71, 72, 129, 248, 253}
# Pure scalar/vector arithmetic only: no calls, branches, images, atomics, extra stores, or pointer conversions.
ALLOWED_OPS |= {20, 41, 42, 44, 46, 79, 80, 81, 82, 83, 113, 114, 124}
ALLOWED_OPS |= set(range(126, 142)) | set(range(149, 153)) | set(range(154, 192)) | set(range(194, 206))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(instructions):
    words = []
    for op, operands in instructions:
        words += [((len(operands) + 1) << 16) | op, *operands]
    return words


def string(words):
    return struct.pack('<' + 'I' * len(words), *words).split(b'\0', 1)[0].decode('utf-8')


def cases():
    """Reuse the fixture owner's public cases, without invoking its translator-running main()."""
    result = fixture.cases()
    require(len(result) == 9 and all(len(row) == 4 and len(row[3]) == 4 for row in result), 'expected nine vec4 cases')
    return result


def adapt(path, destination):
    """Accept only this owned fixture's straight-line, scalar-FP32/vec4-UBO shape."""
    data = path.read_bytes()
    require(20 <= len(data) <= 65536, 'unexpected fixture size')
    header = list(struct.unpack('<5I', data[:20]))
    require(header[0] == 0x07230203 and header[1] <= 0x10300 and header[4] == 0, 'unsupported SPIR-V header')
    ins = fixture.module(path)
    require(all(op in ALLOWED_OPS for op, _ in ins), 'unsupported opcode; not the owned straight-line fixture')
    require([args for op, args in ins if op == 14] == [(0, 1)], 'expected Logical GLSL450 memory model')
    entries = [args for op, args in ins if op == 15]
    require(len(entries) == 1 and entries[0][0] == 4 and string(entries[0][2:4]) == 'main', 'expected one fragment main')
    entry = entries[0]
    main_id = entry[1]
    require(sum(op == 54 for op, _ in ins) == sum(op == 248 for op, _ in ins) == sum(op == 253 for op, _ in ins) == 1,
            'expected one straight-line function/label/return')
    require(all(args[3] in (79, 80, 81) for op, args in ins if op == 12), 'unsupported extended arithmetic')
    require([string(args[1:]) for op, args in ins if op == 11] in ([], ['GLSL.std.450']), 'unexpected extended instructions')
    require(all(args[1] != 40 for op, args in ins if op == 71), 'FPFastMathMode needs a separately reviewed adapter')

    types = {args[0]: (op, args[1:]) for op, args in ins if op in (19, 20, 21, 22, 23, 28, 30, 32, 33)}
    constants = {args[1]: (args[0], args[2:]) for op, args in ins if op == 43}
    variables = {args[1]: (args[0], args[2]) for op, args in ins if op == 59}
    decorations = {}
    offsets = {}
    for op, args in ins:
        if op == 71:
            decorations.setdefault(args[0], {})[args[1]] = args[2:]
        elif op == 72 and args[2] == 35:
            offsets[args[0], args[1]] = args[3]
    output = [ident for ident, (_, storage) in variables.items() if storage == 3]
    require(len(output) == 1 and list(entry[4:]) == output, 'expected exactly one output interface')
    output = output[0]
    pointer_op, (storage, vec_type) = types[variables[output][0]]
    require(pointer_op == 32 and storage == 3 and types[vec_type][0] == 23, 'expected output vector pointer')
    float_type, components = types[vec_type][1]
    require(components == 4 and types[float_type] == (22, (32,)), 'only FP32 vec4 output is supported')
    require(all(args[1] == 32 for op, args in ins if op == 22), 'unexpected floating-point width')
    require(all(args[1] in (32, 64) for op, args in ins if op == 21), 'unexpected integer width')
    needs_int64 = any(args[1] == 64 for op, args in ins if op == 21)

    def integer(ident):
        ty, words = constants[ident]
        require(types[ty][0] == 21 and len(words) == 1, 'nonliteral integer access index')
        return words[0]

    roots = {ident: (ident, ()) for ident in variables}
    active_uniforms, output_components = set(), []
    for op, args in ins:
        if op == 65:
            root, indices = roots[args[2]]
            roots[args[1]] = root, indices + tuple(integer(index) for index in args[3:])
        elif op == 61:
            root, indices = roots[args[2]]
            require(variables[root][1] == 2 and indices in ((0, 0, 0), (0, 0, 1)), 'unexpected input access')
            require(args[0] == float_type, 'unexpected load type')
            active_uniforms.add(root)
        elif op == 62:
            root, indices = roots[args[0]]
            require(root == output and len(indices) == 1, 'unexpected original store')
            output_components += list(indices)
    require(len(active_uniforms) == 1 and sorted(output_components) == [0, 1, 2, 3], 'expected one dynamic UBO and four output stores')
    uniform = active_uniforms.pop()
    descriptor_set = decorations[uniform][34][0]
    input_binding = decorations[uniform][33][0]
    require(descriptor_set == 0, 'runner supports only the fixture descriptor set0')
    all_bindings = [values[33][0] for values in decorations.values() if 33 in values]
    output_binding = max(all_bindings) + 1
    require(output_binding < 32, 'unexpected binding range')
    uniform_struct = types[variables[uniform][0]][1][1]
    require(types[uniform_struct][0] == 30 and len(types[uniform_struct][1]) == 1
            and offsets[uniform_struct, 0] == 0 and 2 in decorations[uniform_struct], 'unexpected uniform block')
    array_type = types[uniform_struct][1][0]
    require(types[array_type][0] == 28 and types[array_type][1][0] == vec_type
            and decorations[array_type][6] == (16,), 'expected std140 vec4 array')
    uniform_bytes = integer(types[array_type][1][1]) * 16
    require(16 <= uniform_bytes <= 65536, 'unexpected uniform range')
    unused = {ident for ident, (_, storage) in variables.items() if storage == 2 and ident != uniform}
    require(len(unused) <= 1 and all(storage in (2, 3) for _, storage in variables.values()), 'unexpected global variables')

    modes = [args for op, args in ins if op == 16]
    require(sum(args == (main_id, 7) for args in modes) == 1, 'expected OriginUpperLeft')
    fp_modes = [args for args in modes if args != (main_id, 7)]
    require(all(len(args) == 3 and args[0] == main_id and args[1] in FP_MODES and args[2] == 32 for args in fp_modes),
            'unknown execution mode; refusing to discard it')
    caps = {args[0] for op, args in ins if op == 17}
    require(caps <= ({1, 11} | UNUSED_CAPABILITIES | set(range(4464, 4469))), 'unknown capability')
    extensions = [string(args) for op, args in ins if op == 10]
    require(set(extensions) <= {'SPV_EXT_fragment_shader_interlock', 'SPV_KHR_float_controls'}, 'unknown extension')

    next_id = header[3]
    struct_id, ptr_struct, result_var, ptr_vec, loaded, address = range(next_id, next_id + 6)
    zero = next(ident for ident, (ty, words) in constants.items() if types[ty][0] == 21 and words == (0,))
    annotations = [(71, (struct_id, 2)), (72, (struct_id, 0, 35, 0)),
                   (71, (result_var, 34, 0)), (71, (result_var, 33, output_binding))]
    declarations = [(30, (struct_id, vec_type)), (32, (ptr_struct, 12, struct_id)),
                    (32, (ptr_vec, 12, vec_type)), (59, (ptr_struct, result_var, 12))]
    copy = [(61, (vec_type, loaded, output)), (65, (ptr_vec, address, result_var, zero)), (62, (address, loaded))]
    adapted, annotated, declared = [], False, False
    for op, args in ins:
        if not annotated and op in (19, 20, 21, 22, 23, 28, 30, 32, 33, 43, 59):
            adapted += annotations
            annotated = True
        if op == 54:
            adapted += declarations
            declared = True
        if op == 17 and args[0] in UNUSED_CAPABILITIES:
            continue  # The whitelist above proves no FP64/image/subgroup/interlock instruction or type is used.
        if op == 10 and string(args) == 'SPV_EXT_fragment_shader_interlock':
            continue
        if op == 59 and args[1] in unused or op in (5, 71) and args[0] in unused:
            continue
        if op == 71 and args[0] == output and args[1] == 30:
            continue
        if op == 15:
            args = (5, main_id, *args[2:4])  # SPIR-V1.3 entry-point interface excludes Private/descriptor variables.
        elif op == 16 and args == (main_id, 7):
            args = (main_id, 17, 1, 1, 1)
        elif op == 32 and args[1] == 3:
            args = (args[0], 6, args[2])
        elif op == 59 and args[2] == 3:
            args = (args[0], args[1], 6, *args[3:])
        if op == 253:
            adapted += copy
        adapted.append((op, args))
    require(annotated and declared, 'adapter insertion points missing')
    source_body = ins[next(i for i, (op, _) in enumerate(ins) if op == 54):]
    result_body = adapted[next(i for i, (op, _) in enumerate(adapted) if op == 54):]
    at_return = next(i for i, (op, _) in enumerate(result_body) if op == 253)
    require(result_body[at_return - 3:at_return] == copy, 'readback copy misplaced')
    require(result_body[:at_return - 3] + result_body[at_return:] == source_body, 'adapter changed original function body')
    header[3] = next_id + 6
    words = header + encode(adapted)
    destination.write_bytes(struct.pack('<' + 'I' * len(words), *words))
    return {'input_set': 0, 'input_binding': input_binding, 'output_binding': output_binding,
            'uniform_bytes': uniform_bytes, 'fp_modes': [list(args[1:]) for args in fp_modes],
            'fp_mode_mask': sum(1 << (args[1] - 4459) for args in fp_modes),
            'requires_shader_int64': needs_int64,
            'removed_unused_capabilities': sorted(caps & UNUSED_CAPABILITIES),
            'source_sha256': hashlib.sha256(data).hexdigest(),
            'adapted_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'unchanged_body_sha256': hashlib.sha256(struct.pack('<' + 'I' * len(encode(source_body)), *encode(source_body))).hexdigest(),
            'float_environment': 'Emitted modes preserved. No float_controls2/FPFastMathDefault; Vulkan defaults apply except listed modes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translator', required=True, type=Path)
    parser.add_argument('--spirv-val', required=True, type=Path)
    parser.add_argument('--cc', default=os.environ.get('CC', 'clang'))
    parser.add_argument('--vulkan-include', required=True, type=Path)
    parser.add_argument('--vulkan-library', required=True, type=Path)
    parser.add_argument('--device-name', default='')
    parser.add_argument('--out', type=Path, default=ROOT / 'build/test/float-minmax-gpu')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--run', action='store_true', help='Explicitly launch Vulkan/GPU work after preparation')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    code = out / 'dynamic.bin'
    code.write_bytes(fixture.shader())
    jobs = out / 'jobs.json'
    jobs.write_text(json.dumps([{'key': 'dynamic', 'stage': 'ps', 'code': str(code.resolve()), 'debug': False}]))
    subprocess.run([str(args.translator.resolve()), str(jobs.resolve()), str(out.resolve())], check=True)
    original, compute = out / 'dynamic.spv', out / 'dynamic-compute.spv'
    subprocess.run([str(args.spirv_val.resolve()), '--target-env', 'vulkan1.2', str(original)], check=True)
    instructions, vectors = fixture.module(original), cases()
    for name, a, b, expected in vectors:
        actual = fixture.evaluate(instructions, [a, b, 0., 0.])
        require(all(math.isnan(got) if math.isnan(want) else fixture.bits(got) == fixture.bits(want)
                    for got, want in zip(actual, expected)), f'owned CPU fixture changed: {name}')
    metadata = adapt(original, compute)
    subprocess.run([str(args.spirv_val.resolve()), '--target-env', 'vulkan1.2', str(compute)], check=True)
    (out / 'adapter.json').write_text(json.dumps(metadata, indent=2) + '\n')
    cc = shutil.which(args.cc) or args.cc
    runner = out / ('float-minmax-gpu.exe' if os.name == 'nt' else 'float-minmax-gpu')
    subprocess.run([str(cc), '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I', str(args.vulkan_include.resolve()), str(ROOT / 'tests/float_minmax_gpu.c'),
                    str(args.vulkan_library.resolve()), '-o', str(runner)], check=True)
    print(f'Prepared/validated original and compute fixture; compiled {runner}. No GPU execution yet.', flush=True)
    if not args.run:
        return 0
    inputs, outputs = out / 'inputs.bin', out / 'outputs.bin'
    inputs.write_bytes(b''.join(struct.pack('<4I', fixture.bits(a), fixture.bits(b), 0, 0) for _, a, b, _ in vectors))
    command = [str(runner), str(compute), str(inputs), str(outputs), str(metadata['input_binding']),
               str(metadata['output_binding']), str(metadata['uniform_bytes']),
               str(metadata['fp_mode_mask'] | (256 if metadata['requires_shader_int64'] else 0)),
               args.device_name, str(int(args.validation))]
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (out / 'runner-stdout.txt').write_text(process.stdout)
    (out / 'runner-stderr.txt').write_text(process.stderr)
    require(process.returncode in (0, 1), f'Vulkan runner failed ({process.returncode}): {process.stderr}')
    # VK_LOADER_DEBUG may add text before our final JSON; keep the full loader log separately.
    device = json.loads(next(line for line in reversed(process.stdout.splitlines()) if line.startswith('{')))
    raw = outputs.read_bytes()
    require(len(raw) == 36 * 4, 'runner did not return all36 values')
    results, mismatches = [], []
    for row, (name, _, _, expected) in enumerate(vectors):
        for column, want in enumerate(expected):
            word = struct.unpack_from('<I', raw, (row * 4 + column) * 4)[0]
            got = fixture.floating(word)
            passed = math.isnan(got) if math.isnan(want) else word == fixture.bits(want)
            result = {'case': name, 'output': column, 'observed_bits': f'{word:08x}',
                      'expected_bits': 'any NaN' if math.isnan(want) else f'{fixture.bits(want):08x}', 'passed': passed}
            results.append(result)
            if not passed:
                mismatches.append(result)
    report = {'device': device, 'adapter': metadata, 'runner_exit': process.returncode,
              'target': 'Observed Maxwell target contract; not Vulkan conformance',
              'checked': len(results), 'mismatches': mismatches, 'results': results}
    (out / 'numerical-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'GPU readback: {len(results)} strict target checks, {len(mismatches)} mismatches; '
          f'Vulkan validation errors {device["validation_errors"]}.')
    for mismatch in mismatches:
        print(json.dumps(mismatch))
    return 1 if mismatches or device['validation_errors'] or process.returncode else 0


if __name__ == '__main__':
    raise SystemExit(main())
