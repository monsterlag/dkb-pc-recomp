"""Translate an owned Maxwell fragment shader and check its floating-point contract.

The inputs are dynamic c[0] loads, so constant folding cannot hide the min/max or SAT
instructions. Same-format F2F.SAT PASS isolates the clamp from upstream arithmetic.
A fourth FP32 output separately exercises constant-folded NaN saturation. FP64
fixtures export the two double results as four raw 32-bit output components.
No game data or GPU is used. The small evaluator executes only this fixture's emitted
SPIR-V subset; it checks the requested numeric contract, not a driver's float controls.

Usage: python tests/float_minmax_test.py --translator PATH [--spirv-val PATH]
"""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def fragment(instructions, fp64=False):
    header = [0] * 20
    header[0] = (5 << 10) | (int(fp64) << 27)  # Fragment stage, optional FP64.
    header[18] = 15      # Render target zero: R0, R1, R2, R3.
    words = []
    for index in range(0, len(instructions), 3):
        words.extend([0, *instructions[index:index + 3]])  # Maxwell scheduling words.
    return struct.pack('<20I', *header) + struct.pack('<' + 'Q' * len(words), *words) + bytes(256)


def shader():

    def ldc(dest, offset):
        return 0xef90000000000000 | (4 << 48) | (offset << 20) | (255 << 8) | (7 << 16) | dest

    def minmax(dest, maximum):
        # FMNMX Rd, R4, R5, PT / !PT. No FTZ, abs/neg or condition-code modifiers.
        return 0x5c60000000000000 | (7 << 39) | (maximum << 42) | (5 << 20) | (4 << 8) | (7 << 16) | dest

    def add_sat(dest, src):
        return 0x5c58000000000000 | (1 << 50) | (255 << 20) | (src << 8) | (7 << 16) | dest

    instructions = [
        ldc(4, 0), ldc(5, 4), minmax(0, 0), minmax(1, 1),
        0x5ca8000000000000 | (1 << 50) | (1 << 39) | (4 << 20) | (2 << 10) | (2 << 8) | (7 << 16) | 2,
        # MOV32I R6, a quiet NaN, followed by FADD.SAT R3, R6, RZ.
        0x0100000000000000 | (0x7fc12345 << 20) | (15 << 12) | (7 << 16) | 6,
        add_sat(3, 6), 0xe30000000007000f,  # EXIT, unconditionally.
    ]
    return fragment(instructions)


def shader_fp64(saturate=False):
    def ldc(dest, offset):
        return 0xef90000000000000 | (5 << 48) | (offset << 20) | (255 << 8) | (7 << 16) | dest

    def minmax(dest, maximum):
        return 0x5c50000000000000 | (7 << 39) | (maximum << 42) | (6 << 20) | (4 << 8) | (7 << 16) | dest

    def sat(dest, src):
        return 0x5ca8000000000000 | (1 << 50) | (1 << 39) | (src << 20) | (3 << 10) | (3 << 8) | (7 << 16) | dest

    operations = [sat(0, 4), sat(2, 6)] if saturate else [minmax(0, 0), minmax(2, 1)]
    return fragment([ldc(4, 0), ldc(6, 8), *operations, 0xe30000000007000f], fp64=True)


def module(path):
    data = path.read_bytes()
    assert len(data) % 4 == 0
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203
    instructions, index = [], 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xffff
        assert count and index + count <= len(words)
        instructions.append((opcode, words[index + 1:index + count]))
        index += count
    return instructions


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def floating(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def float_bits(value, width):
    return int.from_bytes(struct.pack('<f' if width == 32 else '<d', value), 'little')


def bits_float(word, width):
    return struct.unpack('<f' if width == 32 else '<d', word.to_bytes(width // 8, 'little'))[0]


def number_minmax(a, b, maximum):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    if a == b == 0:
        # IEEE numeric min/max: -0 is below +0. No NaN payload promise.
        return floating((bits(a) & bits(b)) if maximum else (bits(a) | bits(b)))
    return max(a, b) if maximum else min(a, b)


def evaluate(instructions, inputs, *, raw=False, raw_outputs=False):
    """Execute the actual emitted straight-line shader using raw scalar bit patterns.

    N-op support is retained for observational comparisons with the earlier translator.
    Integer instructions do not use number_minmax(), which is only the reference for N ops.
    """
    values, types, result_types, pointers, storage, outputs, locals_ = {}, {}, {}, {}, {}, {}, {}
    inputs = inputs if raw else [bits(value) for value in inputs]
    body = False

    def width(ty):
        kind, size, *extra = types[ty]
        return size * width(extra[0]) if kind == 'vector' else size

    def bitcast(value, src_type, dst_type):
        if types[src_type][0] == 'vector':
            lane_width = width(types[src_type][2])
            value = sum(lane << (lane_width * index) for index, lane in enumerate(value))
        if types[dst_type][0] == 'vector':
            count, lane_type = types[dst_type][1:]
            lane_width = width(lane_type)
            return [(value >> (lane_width * index)) & ((1 << lane_width) - 1) for index in range(count)]
        assert width(src_type) == width(dst_type)
        return value

    for opcode, args in instructions:
        if opcode == 20:
            types[args[0]] = ('bool', 1)
        elif opcode in (21, 22):
            types[args[0]] = ('float' if opcode == 22 else 'int', args[1])
        elif opcode == 23:
            types[args[0]] = ('vector', args[2], args[1])
        elif opcode == 43:  # OpConstant, literal words remain unchanged.
            ty, result, *words = args
            result_types[result] = ty
            values[result] = sum(word << (32 * index) for index, word in enumerate(words))
        elif opcode == 59:  # OpVariable
            _, result, kind = args[:3]
            pointers[result] = (result, ())
            storage[result] = kind
        elif opcode == 65:  # OpAccessChain
            _, result, base, *indices = args
            root, path = pointers[base]
            pointers[result] = (root, path + tuple(values[i] for i in indices))
        elif opcode == 61:  # OpLoad: c[0].data[vec4][component].
            ty, result, pointer = args
            root, path = pointers[pointer]
            result_types[result] = ty
            if storage[root] == 7:
                assert not path
                values[result] = locals_[root]
            else:
                assert storage[root] == 2 and len(path) == 3 and path[0] == 0
                values[result] = inputs[path[1] * 4 + path[2]]
        elif opcode in (80, 81, 124, 129, 169, 170, 171, 172, 176, 198, 199, 200, 12):
            ty, result, *operands = args
            result_types[result] = ty
            if opcode == 80:  # OpCompositeConstruct
                values[result] = [values[i] for i in operands]
            elif opcode == 81:  # OpCompositeExtract
                value = values[operands[0]]
                for index in operands[1:]:
                    value = value[index]
                values[result] = value
            elif opcode == 124:  # OpBitcast
                source = operands[0]
                values[result] = bitcast(values[source], result_types[source], ty)
            elif opcode == 129:  # OpFAdd: only retained for old-fixture compatibility.
                a, b = [bits_float(values[i], width(ty)) for i in operands]
                values[result] = float_bits(a + b, width(ty))
            elif opcode == 169:  # OpSelect, with integer result in the fixed operations.
                condition, a, b = operands
                values[result] = values[a] if values[condition] else values[b]
            elif opcode == 12:
                _, operation, *ids = operands
                if operation == 59:  # PackDouble2x32 is a bit encoding, not a conversion.
                    low, high = values[ids[0]]
                    values[result] = low | (high << 32)
                elif operation == 65:  # UnpackDouble2x32
                    value = values[ids[0]]
                    values[result] = [value & 0xffffffff, value >> 32]
                else:
                    data = [bits_float(values[i], width(ty)) for i in ids]
                    if operation in (79, 80):
                        value = number_minmax(*data, operation == 80)
                    elif operation == 81:
                        x, low, high = data
                        value = number_minmax(number_minmax(x, low, True), high, False)
                    else:
                        raise AssertionError(f'unsupported GLSL.std.450 operation {operation}')
                    values[result] = float_bits(value, width(ty))
            else:
                data = [values[i] for i in operands]
                if opcode == 170:
                    value = data[0] == data[1]
                elif opcode == 171:
                    value = data[0] != data[1]
                elif opcode == 172:
                    value = data[0] > data[1]
                elif opcode == 176:
                    value = data[0] < data[1]
                elif opcode == 198:
                    value = data[0] ^ data[1]
                elif opcode == 199:
                    value = data[0] & data[1]
                else:
                    value = ~data[0] & ((1 << width(ty)) - 1)
                values[result] = value
        elif opcode == 62:  # OpStore to fragment output components.
            pointer, value = args
            root, path = pointers[pointer]
            if storage[root] == 7:
                assert not path
                locals_[root] = values[value]
            else:
                assert storage[root] == 3 and len(path) == 1
                outputs[path[0]] = values[value]
        elif opcode == 54:
            body = True
        elif opcode == 56:
            body = False
        elif body and opcode not in (248, 253):  # OpLabel, OpReturn.
            raise AssertionError(f'unsupported shader-body opcode {opcode}')
    assert set(outputs) == {0, 1, 2, 3}
    result = [outputs[i] for i in range(4)]
    return result if raw_outputs else [floating(word) for word in result]


def cases():
    """Shared FP32 target-contract cases; the GPU probe reports host observations separately."""
    return [
        ('finite', 2., -3., [-3., 2., 1., 0.]),
        ('NaN first', math.nan, 1., [1., 1., 0., 0.]),
        ('NaN second', 1., math.nan, [1., 1., 1., 0.]),
        ('both NaN', math.nan, math.nan, [math.nan, math.nan, 0., 0.]),
        ('infinities', -math.inf, math.inf, [-math.inf, math.inf, 0., 0.]),
        ('infinities reversed', math.inf, -math.inf, [-math.inf, math.inf, 1., 0.]),
        ('negative zero first', -0., 0., [-0., 0., 0., 0.]),
        ('negative zero second', 0., -0., [-0., 0., 0., 0.]),
        ('NaN and negative zero', math.nan, -0., [-0., -0., 0., 0.]),
    ]


def check_integer_selection(instructions):
    operations = [args[3] for opcode, args in instructions if opcode == 12]
    assert not set(operations) & {37, 40, 43, 79, 80, 81}, f'floating min/max/clamp retains implicit special-value assumptions: {operations}'
    assert {124, 169, 172, 176, 198, 199, 200} <= {op for op, _ in instructions}
    bitcasts = {args[1] for op, args in instructions if op == 124}
    precise = {args[0] for op, args in instructions if op == 71 and args[1] == 42}
    assert not bitcasts & precise, 'NoContraction cannot decorate the integer helper result bitcast'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translator', required=True, type=Path)
    parser.add_argument('--spirv-val', type=Path)
    args = parser.parse_args()
    out = ROOT / 'build/test/float-minmax'
    out.mkdir(parents=True, exist_ok=True)
    fixtures = {'dynamic': shader(), 'dynamic64': shader_fp64(), 'dynamic64_sat': shader_fp64(saturate=True)}
    jobs = []
    for name, data in fixtures.items():
        code = out / (name + '.bin')
        code.write_bytes(data)
        jobs.append({'key': name, 'stage': 'ps', 'code': str(code.resolve()), 'debug': False})
    job_file = out / 'jobs.json'
    job_file.write_text(json.dumps(jobs))
    result = subprocess.run([str(args.translator.resolve()), str(job_file.resolve()), str(out.resolve())],
                            capture_output=True, encoding='utf-8')
    if result.returncode:
        raise AssertionError(f'translator failed: {result.stderr.encode("ascii", "backslashreplace").decode()}')
    modules = {}
    for name in fixtures:
        spirv = out / (name + '.spv')
        if args.spirv_val:
            subprocess.run([str(args.spirv_val.resolve()), '--target-env', 'vulkan1.2', str(spirv)], check=True)
        modules[name] = module(spirv)
        check_integer_selection(modules[name])
    instructions = modules['dynamic']
    assert sum(opcode == 61 for opcode, _ in instructions) >= 3, 'fixture lost its dynamic loads'
    vectors = cases() + [
        ('fractions', .5, .25, [.25, .5, .5, 0.]),
        ('positive subnormals', floating(1), floating(2), [floating(1), floating(2), floating(1), 0.]),
        ('negative subnormals', floating(0x80000001), floating(0x80000002),
         [floating(0x80000002), floating(0x80000001), 0., 0.]),
        ('both negative zero', -0., -0., [-0., -0., 0., 0.]),
    ]
    checked32 = 0
    for name, a, b, expected in vectors:
        actual = evaluate(instructions, [a, b, 0., 0.])
        for index, (got, want) in enumerate(zip(actual, expected)):
            correct = math.isnan(got) if math.isnan(want) else bits(got) == bits(want)
            assert correct, (name, index, got, want)
            checked32 += 1

    # Cross product covers both NaN orders/pairs, both zero signs, equal operands,
    # infinities, smallest/largest subnormals, normal boundaries and finite extremes.
    patterns = [0, 0x8000000000000000, 1, 0x8000000000000001,
                0x000fffffffffffff, 0x800fffffffffffff, 0x0010000000000000, 0x8010000000000000,
                0x3fe0000000000000, 0xbfe0000000000000, 0x7fefffffffffffff, 0xffefffffffffffff,
                0x7ff0000000000000, 0xfff0000000000000, 0x7ff8123456789abc, 0xfff89abcdef01234]
    checked64 = 0
    for a_bits in patterns:
        for b_bits in patterns:
            a, b = bits_float(a_bits, 64), bits_float(b_bits, 64)
            words = [a_bits & 0xffffffff, a_bits >> 32, b_bits & 0xffffffff, b_bits >> 32]
            for name in ('dynamic64', 'dynamic64_sat'):
                result = evaluate(modules[name], words, raw=True, raw_outputs=True)
                actual = [result[0] | (result[1] << 32), result[2] | (result[3] << 32)]
                if name == 'dynamic64':
                    expected = [number_minmax(a, b, False), number_minmax(a, b, True)]
                else:
                    expected = [0. if math.isnan(x) or x <= 0. else 1. if x >= 1. else x for x in (a, b)]
                for index, (got, want) in enumerate(zip(actual, expected)):
                    correct = math.isnan(bits_float(got, 64)) if math.isnan(want) else got == float_bits(want, 64)
                    assert correct, (name, hex(a_bits), hex(b_bits), index, hex(got), want)
                    checked64 += 1
    assert any(op == 17 and arg == (11,) for op, arg in modules['dynamic64']), 'missing Int64 capability'
    assert not any(op == 17 and arg == (11,) for op, arg in instructions), 'FP32 unexpectedly requires Int64'
    print(f'float min/max: {checked32} FP32 + {checked64} FP64 emitted-SPIR-V value checks passed')
    print('Integer selection at the operation boundary; constant SAT NaN -> +0. CPU interpretation only.')
    print('Upstream arithmetic, FTZ modifiers, signaling-NaN/payload fidelity and GPU execution remain separate.')


if __name__ == '__main__':
    main()
