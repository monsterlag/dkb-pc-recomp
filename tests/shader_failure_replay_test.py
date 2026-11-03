"""Exercise actual failed-request serialization/cleanup, then optional offline translator replay.

All payloads are original synthetic bytes. No GPU, game, or .NET build is used.
"""
import argparse
import importlib.util
import json
import os
import queue
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
CC = None
TRANSLATOR = None


def function(text, name):
    match = re.search(r'^(?:static )?[^\n;{}]+\b' + name + r'\([^;{}]*\)\s*\{', text, re.M)
    if not match:
        raise ValueError(name)
    depth = 0
    for offset in range(text.index('{', match.start()), len(text)):
        depth += (text[offset] == '{') - (text[offset] == '}')
        if not depth:
            return text[match.start():offset + 1]
    raise ValueError(name)


def record(text, name):
    end = text.index('} ' + name + ';') + len('} ' + name + ';')
    return text[text.rfind('typedef struct {', 0, end):end]


class ShaderFailureCapture(unittest.TestCase):
    def packet_request(self, root):
        requests = list(root.rglob('*.request.json'))
        self.assertEqual(len(requests), 1, 'one complete captured request expected')
        return requests[0]

    @classmethod
    def setUpClass(cls):
        output = ROOT / 'build/test/shader-failure-replay'
        output.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=output)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        shader = (ROOT / 'runtime/gfx_shader.c').read_text()
        header = (ROOT / 'runtime/gfx.h').read_text()
        functions = [function(shader, name) for name in ('json_esc', 'key_hex', 'nvn2_shared', 'attr_name', 'build_request')]
        if 'static void dump_failed_request(' in shader:
            functions.append(function(shader, 'dump_failed_request'))
        functions.append(function(shader, 'translate_one'))
        names = []
        for name in ('tex_names', 'fmt_names'):
            start = shader.index('static const char *const ' + name)
            names.append(shader[start:shader.index(';', start) + 1])
        source = (ROOT / 'tests/shader_failure_capture.c').read_text()
        types = [record(text, name) for text, name in
                 ((header, 'GfxShaderAnswers'), (shader, 'UseMask'), (shader, 'Req'))]
        types.append(re.search(r'^enum \{ GFX_STORAGE_CANDIDATES_VERSION = [^\n]+;', header, re.M).group())
        source = source.replace('@TYPES@', '\n'.join(types))
        source = source.replace('@FUNCTIONS@', '\n'.join(names + functions))
        generated = cls.work / 'fixture.c'
        generated.write_text(source)
        cls.exe = cls.work / ('fixture.exe' if os.name == 'nt' else 'fixture')
        subprocess.run([CC, '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'runtime'),
                        str(generated), '-o', str(cls.exe)],
                       check=True, timeout=30)

    def capture(self, enabled, mode=0, stage=4):
        work = Path(tempfile.mkdtemp(dir=self.work))
        cache, packet = work / 'cache', work / '[inputs] & spaces'
        cache.mkdir()
        packet.mkdir()
        env = dict(os.environ)
        env.pop('DKB_SHADER_FAILURE_DUMP', None)
        if enabled:
            env['DKB_SHADER_FAILURE_DUMP'] = str(packet)
        run = subprocess.run([str(self.exe), str(cache), str(mode), str(stage)], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        return cache, packet, run.stdout

    def test_default_failure_does_not_capture(self):
        cache, packet, text = self.capture(False)
        self.assertEqual(list(packet.iterdir()), [])
        self.assertIn('writes=0', text)
        self.assertTrue((cache / 'sent.json').is_file())

    def test_opt_in_keeps_exact_request_and_lowered_bytes(self):
        for stage in (4, 5):
            with self.subTest(stage=stage):
                cache, packet, text = self.capture(True, stage=stage)
                key = 'ab' * 32
                request = self.packet_request(packet)
                self.assertTrue(request.is_file(), 'failed request was discarded')
                self.assertEqual(request.read_bytes(), (cache / 'sent.json').read_bytes())
                self.assertEqual(request.with_name(key + '.failed.code').read_bytes(), bytes([1, 2, 3, 4]))
                self.assertEqual(request.with_name(key + '.error.txt').read_text(), 'fail synthetic')
                job = json.loads(request.read_text())['job']
                self.assertEqual(job['c1'], 'abcdef')
                self.assertEqual(job['tex'], {'3': '2d'})
                self.assertEqual(job['shared'], 128)
                self.assertTrue(job['softfloat64'])
                self.assertEqual(job['sbuf_align'], 16)
                if stage == 4:
                    self.assertTrue(job['early_fragment_tests'])
                    self.assertEqual(job['subgroup'], 64)
                    for key in ('c0', 'c0_version', 'c0_roots', 'storage_candidates_version'):
                        self.assertNotIn(key, job)
                else:
                    self.assertEqual(job['local'], [4, 2, 1])
                    self.assertEqual(job['c0'], [1, 2, 3, 4])
                    self.assertEqual(job['c0_version'], 2)
                    self.assertEqual(job['storage_candidates_version'], 1)
                    self.assertEqual(job['c0_roots'], [1, 0, 7, 4, 64, 2, 0, 9])
                self.assertIn('writes=3', text)

    def test_success_and_transport_failure_do_not_capture(self):
        for mode in (1, 2):
            for stage in (4, 5):
                with self.subTest(mode=mode, stage=stage):
                    _, packet, text = self.capture(True, mode=mode, stage=stage)
                    self.assertEqual(list(packet.iterdir()), [])
                    self.assertIn('writes=0', text)

    def check_capture_in_cache(self, alias):
        cache, _, _ = self.capture(False)
        directory = cache if alias == 'exact' else cache / '..' / cache.name
        if alias == 'case':
            directory = cache.with_name(cache.name.upper())
        env = dict(os.environ, DKB_SHADER_FAILURE_DUMP=str(directory))
        run = subprocess.run([str(self.exe), str(cache), '0', '4'], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        key = 'ab' * 32
        request = self.packet_request(cache)
        self.assertEqual(request.with_name(key + '.error.txt').read_text(), 'fail synthetic')
        self.assertFalse((cache / (key + '.code')).exists(), 'normal temporary code cleanup must remain enabled')
        binary = request.with_name(key + '.failed.code')
        self.assertTrue(binary.is_file(), 'published repro packet lost its captured code during cache cleanup')
        self.assertEqual(binary.read_bytes(), bytes([1, 2, 3, 4]))

    def test_dump_may_share_the_cache_directory(self):
        self.check_capture_in_cache('exact')

    def test_dump_may_alias_the_cache_directory(self):
        for alias in (('parent', 'case') if os.name == 'nt' else ('parent',)):
            with self.subTest(alias=alias):
                self.check_capture_in_cache(alias)

    def test_first_complete_packet_is_preserved(self):
        cache, packet, _ = self.capture(True)
        request = self.packet_request(packet)
        original = request.read_bytes()
        binary = request.with_name('ab' * 32 + '.failed.code')
        binary.write_bytes(b'previous capture')
        env = dict(os.environ, DKB_SHADER_FAILURE_DUMP=str(packet))
        run = subprocess.run([str(self.exe), str(cache), '0', '5'], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('writes=3', run.stdout)
        self.assertEqual(request.read_bytes(), original)
        self.assertEqual(binary.read_bytes(), b'previous capture')
        self.assertEqual(len(list(packet.rglob('*.request.json'))), 2, 'later capture uses independent packet storage')

    def test_two_writers_cannot_replace_a_published_packet(self):
        shared = Path(tempfile.mkdtemp(dir=self.work))
        caches = [Path(tempfile.mkdtemp(dir=self.work)) for _ in range(2)]
        env = dict(os.environ, DKB_SHADER_FAILURE_DUMP=str(shared), FIXTURE_CAPTURE_GATE='1', FIXTURE_CAPTURE_PID='17')
        children = []
        try:
            for writer, cache in enumerate(caches, 1):
                child = subprocess.Popen([str(self.exe), str(cache), '0', '4', str(writer)], env=env,
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                children.append(child)
                ready = queue.Queue()
                threading.Thread(target=lambda p=child, q=ready: q.put(p.stdout.readline()), daemon=True).start()
                self.assertEqual(ready.get(timeout=10), 'capture write ready\n')
            # Both passed the directory/file publication checks and are latched before their first write.
            # Finish writer one completely, then let writer two perform all of its writes.
            text, error = children[0].communicate('x', timeout=10)
            self.assertEqual(children[0].returncode, 0, text + error)
            first = self.packet_request(shared)
            snapshot = {p: p.read_bytes() for p in first.parent.glob(first.stem.removesuffix('.request') + '.*')}
            text, error = children[1].communicate('x', timeout=10)
            self.assertEqual(children[1].returncode, 0, text + error)
            self.assertEqual({p: p.read_bytes() for p in snapshot}, snapshot, 'second writer replaced a published packet')
            requests = list(shared.rglob('*.request.json'))
            self.assertEqual(len(requests), 2)
            self.assertNotEqual(requests[0].parent, requests[1].parent)
            for request in requests:
                job = json.loads(request.read_text())['job']
                writer = job['gather_bias']
                self.assertIn(writer, (1, 2))
                self.assertEqual(request.with_name('ab' * 32 + '.failed.code').read_bytes(), bytes([writer, 2, 3, 4]))
                self.assertEqual(request.with_name('ab' * 32 + '.error.txt').read_text(), f'fail synthetic writer {writer}')
                self.assertTrue(request.parent.name.endswith(f'-17-{writer}'), 'atomic claim must handle a reused process id')
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=10)

    def test_capture_io_failure_preserves_original_failure_and_cleanup(self):
        cache, packet, _ = self.capture(False)
        env = dict(os.environ, DKB_SHADER_FAILURE_DUMP=str(packet / 'missing-parent' / 'missing-child'))
        run = subprocess.run([str(self.exe), str(cache), '0', '4'], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(any(cache.glob('*.code')))
        self.assertEqual(list(packet.iterdir()), [])

    def test_partial_packet_is_not_published_or_reused(self):
        cache, packet, _ = self.capture(False)
        env = dict(os.environ, DKB_SHADER_FAILURE_DUMP=str(packet), FIXTURE_CAPTURE_WRITE_FAIL_AT='2',
                   FIXTURE_CAPTURE_PID='17')
        run = subprocess.run([str(self.exe), str(cache), '0', '4'], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertFalse(any(packet.rglob('*.request.json')), 'incomplete capture must not publish a JSON packet')
        old = next(packet.glob('capture-17-1'))
        self.assertEqual((old / ('ab' * 32 + '.failed.code')).read_bytes(), bytes([1, 2, 3, 4]))
        snapshot = {p: p.read_bytes() for p in old.iterdir()}
        env.pop('FIXTURE_CAPTURE_WRITE_FAIL_AT')
        run = subprocess.run([str(self.exe), str(cache), '0', '4'], env=env,
                             capture_output=True, text=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.packet_request(packet).parent.name, 'capture-17-2')
        self.assertEqual({p: p.read_bytes() for p in old.iterdir()}, snapshot)

    def test_actual_translator_replays_saved_failure_with_stack(self):
        if TRANSLATOR is None:
            self.skipTest('pass --translator for actual offline replay')
        cache, packet, _ = self.capture(True)
        request = self.packet_request(packet)
        output = packet.parent / 'replay output'
        command = [sys.executable, '-B', str(ROOT / 'tools/replay_shader_failure.py'), str(request),
                   '--translator', str(TRANSLATOR), '--out', str(output)]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
        self.assertFalse(any(cache.glob('*.code')), 'replay must use retained sibling bytes')
        self.assertIn('fail ' + 'ab' * 32, (output / 'reply.txt').read_text())
        self.assertIn(' at ', (output / 'stderr.txt').read_text())

    def test_actual_translator_success_does_not_modify_packet(self):
        if TRANSLATOR is None:
            self.skipTest('pass --translator for actual offline replay')
        _, packet, _ = self.capture(True)
        request = self.packet_request(packet)
        original = request.read_bytes()
        owned = runpy.run_path(str(ROOT / 'tools/xlate_smoke.py'))['fragment_code']()
        binary = request.with_name('ab' * 32 + '.failed.code')
        binary.write_bytes(owned)
        output = packet.parent / 'successful replay'
        run = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/replay_shader_failure.py'), str(request),
                              '--translator', str(TRANSLATOR), '--out', str(output)],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(request.read_bytes(), original)
        self.assertEqual(binary.read_bytes(), owned)
        spirv = output / ('ab' * 32 + '.spv')
        self.assertEqual(spirv.read_bytes()[:4], b'\x03\x02\x23\x07')

    def test_replay_requires_sibling_code_and_matching_safe_key(self):
        _, packet, _ = self.capture(True)
        spec = importlib.util.spec_from_file_location('replay_failure', ROOT / 'tools/replay_shader_failure.py')
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        request = self.packet_request(packet)
        body = json.loads(request.read_text())
        body['job']['key'] = '../outside'
        request.write_text(json.dumps(body))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            replay.replay(request, Path('unused'), packet / 'output', 1)
        body['job']['key'] = 'ab' * 32
        request.write_text(json.dumps(body))
        request.with_name('ab' * 32 + '.failed.code').unlink()
        with self.assertRaisesRegex(ValueError, 'sibling .failed.code file is missing'):
            replay.replay(request, Path('unused'), packet / 'output', 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('cc'))
    parser.add_argument('--translator', type=Path)
    arguments, remaining = parser.parse_known_args()
    if not arguments.cc:
        parser.error('pass --cc')
    CC, TRANSLATOR = arguments.cc, arguments.translator
    unittest.main(argv=[sys.argv[0], *remaining])
