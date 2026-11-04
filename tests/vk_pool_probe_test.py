"""Prepare/check the original vk_pool_probe Maxwell shader without loading Vulkan.

The GPU check is the runtime's --probe-vkpool mode. This script checks the exact
in-source instructions, translator reflection and SPIR-V; it does not prove a
driver result or descriptor-pool switching by itself.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


def shader_words():
    source = (ROOT / 'runtime/vk_pool_probe.c').read_text()
    array = source.split('static const uint64_t pool_probe_code[] = {', 1)[1].split('};', 1)[0]
    return [int(x, 16) for x in re.findall(r'UINT64_C\(0x([0-9a-f]+)\)', array)]


def spv_instructions(data):
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203
    offset = 5
    while offset < len(words):
        size, opcode = words[offset] >> 16, words[offset] & 65535
        assert size and offset + size <= len(words)
        yield opcode, words[offset + 1:offset + size]
        offset += size


class ProbePreparation(unittest.TestCase):
    def test_original_instruction_fields(self):
        words = shader_words()
        self.assertEqual(len(words), 12)
        self.assertEqual(words[::4], [0x001f8000fc0007e0] * 3)
        ops = [w for index, w in enumerate(words) if index % 4]
        for op, destination, bits in zip(ops[:2], (0, 1), (0x3fa00000, 0x3f000000)):
            self.assertEqual(op >> 56, 1)  # MOV32I
            self.assertEqual(op & 255, destination)
            self.assertEqual((op >> 20) & 0xffffffff, bits)
            self.assertEqual((op >> 16) & 7, 7)
        for op, destination, word in ((ops[2], 2, 8), (ops[4], 4, 4), (ops[5], 5, 5)):
            self.assertEqual(op >> 51, 0x4c98000000000000 >> 51)
            self.assertEqual(op & 255, destination)
            self.assertEqual((op >> 34) & 31, 0)
            self.assertEqual((op >> 20) & 0x3fff, word)
        tex = ops[3]
        self.assertEqual(tex >> 54, int('1101111010', 2))  # InstTable: TEX.B
        self.assertEqual((tex >> 28) & 7, 2)  # 2D
        self.assertEqual((tex >> 37) & 7, 1)  # LZ
        self.assertEqual((tex >> 31) & 15, 15)
        self.assertEqual((tex & 255, (tex >> 8) & 255, (tex >> 20) & 255), (8, 0, 2))
        store = ops[6]
        self.assertEqual(store >> 51, 0xeed8000000000000 >> 51)
        self.assertEqual((store >> 48) & 7, 6)  # B128
        self.assertEqual((store & 255, (store >> 8) & 255, (store >> 20) & 0xffffff), (8, 4, 0))
        self.assertTrue(store & (1 << 45))  # extended 64-bit address
        self.assertEqual(ops[7], 0xe30000000007000f)
        self.assertEqual(ops[8], 0)

    def test_expected_nearest_samples(self):
        # An independent integer/coordinate oracle for the original authored pattern:
        # u=1.25, repeat -> .25 -> texel0; clamp -> 1 -> final texel1.
        red, green = (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)
        def nearest(pixels, repeat):
            coordinate = 1.25 % 1 if repeat else min(1, max(0, 1.25))
            return pixels[min(len(pixels) - 1, int(coordinate * len(pixels)))]
        self.assertEqual([nearest(p, True) for p in ((red, red), (green, green), (red, red))], [red, green, red])
        self.assertEqual([nearest((red, green), repeat) for repeat in (True, False, True)], [red, green, red])
        self.assertEqual(struct.pack('<4f', *red), struct.pack('<4I', 0x3f800000, 0, 0, 0x3f800000))
        self.assertEqual(struct.pack('<4f', *green), struct.pack('<4I', 0, 0x3f800000, 0, 0x3f800000))

    def test_actual_translator_and_spirv(self):
        args.out.mkdir(parents=True, exist_ok=True)
        code = struct.pack('<' + 'Q' * len(shader_words()), *shader_words())
        source = args.out / 'original.bin'
        source.write_bytes(code)
        jobs = []
        for key, alignment in (('exact', None), ('aligned256', 256)):
            job = dict(key=key, stage='cs', code=str(source), debug=False, c1='', local=[1, 1, 1], shared=0,
                       c0=[0, 0, 1, 1], c0_version=2, c0_roots=[0] * 8, tex={'0': '2d'})
            if alignment is not None:
                job['sbuf_align'] = alignment
            jobs.append(job)
        env = {k: v for k, v in os.environ.items() if k.upper() in {
            'SYSTEMROOT', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA',
            'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMDATA', 'SYSTEMDRIVE'}}
        run = subprocess.run([str(args.xlate.resolve()), '--serve'], env=env,
                             input=''.join(json.dumps(dict(job=j, out=str(args.out.resolve()))) + '\n' for j in jobs),
                             text=True, capture_output=True, timeout=60)
        (args.out / 'translator.log').write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines(), ['ok exact', 'ok aligned256'], run.stdout + run.stderr)
        results = []
        for job in jobs:
            spv = args.out / (job['key'] + '.spv')
            reflection = json.loads((args.out / (job['key'] + '.res.json')).read_text())
            self.assertEqual(len(reflection['sbufs']), 1)
            buffer = reflection['sbufs'][0]
            self.assertEqual((buffer['cb_slot'], buffer['cb_offset'], buffer['flags']), (0, 4, 1))
            self.assertEqual(len(reflection['textures']), 1)
            texture = reflection['textures'][0]
            self.assertEqual((texture['cbuf_slot'], texture['handle'], texture['type'], texture['separate']), (0, 8, 3, False))
            # TEX.B.2D encodes its type. BindlessElimination only asks the host
            # for ambiguous buffer types and queries; type-cache coverage is a
            # separate CPU test, not a claim made by this same-type GPU probe.
            self.assertEqual(reflection['tex_queried'], [])
            self.assertEqual(reflection['images'], [])
            self.assertEqual(reflection['notes'], [])
            validation = subprocess.run([str(args.spirv_val.resolve()), '--target-env', 'vulkan1.3', str(spv.resolve())],
                                        text=True, capture_output=True, timeout=15)
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
            instructions = list(spv_instructions(spv.read_bytes()))
            self.assertEqual(sum(op == 88 for op, _ in instructions), 1)  # OpImageSampleExplicitLod
            decorations = {o[0]: o[2] for op, o in instructions if op == 71 and o[1] == 33}
            storage = {var for var, binding in decorations.items() if binding == buffer['binding']}
            pointers = {o[1]: o[2] for op, o in instructions if op == 65}
            self.assertEqual(sum(op == 62 and pointers.get(o[0]) in storage for op, o in instructions), 4)
            results.append(dict(key=job['key'], spirv_sha256=hashlib.sha256(spv.read_bytes()).hexdigest(),
                                reflection=reflection, validation='PASS'))
        report = dict(source_sha256=hashlib.sha256((ROOT / 'runtime/vk_pool_probe.c').read_bytes()).hexdigest(),
                      code_sha256=hashlib.sha256(code).hexdigest(),
                      translator_sha256=hashlib.sha256(args.xlate.read_bytes()).hexdigest(), results=results,
                      gpu_executed=False, note='Compiler preparation only; actual pool-selection readback remains required.')
        (args.out / 'preparation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xlate', required=True, type=Path)
    parser.add_argument('--spirv-val', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args, remaining = parser.parse_known_args()
    unittest.main(argv=[__file__] + remaining)
