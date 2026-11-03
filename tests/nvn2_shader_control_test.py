"""Run the real shader-request guards against synthetic short/old/new controls, without a GPU or translator.

The short control ends at a protected page: reading the NVN 1 layout before rejecting NVN 2 must fault.
Usage: python tests/nvn2_shader_control_test.py [--cc clang]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from nvn2_compute_c0_test import function

ROOT = Path(__file__).resolve().parents[1]


def source():
    production = (ROOT / 'runtime' / 'gfx_shader.c').read_text()
    # Stub only translation/cache dependencies reached after the real request guards.
    before = r'''
#include <stdint.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn2_c0.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#define NVN2_C0_LAYOUT 0x860
#define CODE_SKIP 0x30
enum { GFX_STAGE_VS, GFX_STAGE_FS=4, GFX_STAGE_CS };
typedef int GfxShaderAnswers;
typedef int UseMask;
typedef struct { const char *stage_name; uint32_t length, c1_len; const uint8_t *c1;
                 uint8_t code_hash[32], key[32], iid[32]; int final, require_c0; } Req;
enum { GFX_RES_KINDS=4 };
typedef struct { char *reflection; int stage, has_c0; Nvn2C0Layout c0; int nres[4]; int *res[4]; } GfxShader;
static void rt_log(const char *message) { (void)message; }
static void add_res(GfxShader *s, int kind, const char *o, const char *e) { (void)s; (void)kind; (void)o; (void)e; }
static void reflect_spirv(GfxShader *s) { (void)s; }
static const char *error;
static int fail(const char *why, const char *detail) { (void)detail; error = why; return 0; }
static int build_code_hash(int stage, const uint8_t *control, const uint8_t *code, uint64_t length,
                           const uint8_t *c1, uint32_t c1_len, uint8_t *hash) {
    (void)stage; (void)control; (void)code; (void)length; (void)c1; (void)c1_len; (void)hash; return 1;
}
static void paths(void) {}
static int use_find(const uint8_t *hash, UseMask *mask) { (void)hash; (void)mask; return 0; }
static void build_key(const uint8_t *hash, const GfxShaderAnswers *answers, const UseMask *mask,
                      uint8_t *key, uint8_t *iid) { (void)hash; (void)answers; (void)mask; (void)key; (void)iid; }
'''
    after = r'''
int main(void) {
    size_t page;
    uint8_t *mapping;
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); page = si.dwPageSize;
    mapping = VirtualAlloc(NULL, page * 2, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    DWORD old;
    if (!mapping || !VirtualProtect(mapping + page, page, PAGE_NOACCESS, &old)) return 2;
#else
    page = (size_t)sysconf(_SC_PAGESIZE);
    mapping = mmap(NULL, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED || mprotect(mapping + page, page, PROT_NONE)) return 2;
#endif
    uint8_t *short_control = mapping + page - 0xd8;
    const uint32_t header[2] = {0x98761234u, 0xc9u};
    memcpy(short_control, header, sizeof header);
    uint8_t code[8] = {0}, full[0x880] = {0};
    Req req = {0};
    GfxShaderAnswers answers = 0;
    int checks = 0;
#define CHECK(c) do { checks++; if (!(c)) { fprintf(stderr, "shader control: failed check %d\n", checks); return 1; } } while (0)
    CHECK(!req_init(&req, GFX_STAGE_CS, short_control, code, sizeof code, NULL, &answers) &&
          error && strstr(error, "NVN 2 program"));
    memcpy(full + NVN2_C0_LAYOUT, "DKC0", 4);
    error = NULL;
    CHECK(!req_init(&req, GFX_STAGE_CS, full, code, sizeof code, NULL, &answers) &&
          error && strstr(error, "old NVN 2 compute control"));
    const uint32_t length = sizeof code;
    memcpy(full + 0x700, &length, 4); memcpy(full + 0x704, &length, 4);
    memcpy(full + NVN2_C0_LAYOUT, "DKC1", 4);
    CHECK(req_init(&req, GFX_STAGE_CS, full, code, sizeof code, NULL, &answers) && req.require_c0);
    memset(full + NVN2_C0_LAYOUT, 0, 4);
    CHECK(req_init(&req, GFX_STAGE_CS, full, code, sizeof code, NULL, &answers) && !req.require_c0);
#ifdef _WIN32
    VirtualFree(mapping, 0, MEM_RELEASE);
#else
    munmap(mapping, page * 2);
#endif
    const char *profiles[] = {
        "{\"stage\":\"cs\"}",
        "{\"stage\":\"cs\"}",
        "{\"stage\":\"cs\",\"c0\":[7,1,2,3]}",
        "{\"stage\":\"cs\",\"c0_version\":2,\"c0\":[7,1,2,3],\"c0_roots\":[3,0,5,0,0,0,0,0]}",
        "{\"stage\":\"cs\",\"c0_version\":2,\"c0\":[7,1,2,3],\"c0_roots\":[65,0,0,0,0,0,0,0]}"
    };
    const int require[] = {1,0,1,1,1}, expected[] = {0,1,0,1,0};
    for (unsigned i = 0; i < sizeof profiles / sizeof *profiles; i++) {
        GfxShader s = {.reflection=(char *)profiles[i]};
        int ok = reflect(&s, require[i]);
        CHECK(ok == expected[i] && (i != 3 || (s.has_c0 && s.c0.roots[0] == 3 && s.c0.roots[2] == 5)));
        for (int k = 0; k < 4; k++) free(s.res[k]);
    }
    printf("shader control/profile: %d checks, 0 failures\n", checks);
    return 0;
}
'''
    pieces = [function(production, name) for name in (
        'static const char *json_key(', 'static long json_int(', 'static void json_str(',
        'static int json_objects(', 'int gfx_shader_is_nvn2(', 'static int req_init(')]
    pieces.append(function(production[production.rindex('static int reflect('):], 'static int reflect('))
    return before + '\n'.join(pieces) + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc', default=os.environ.get('CC') or shutil.which('clang') or shutil.which('gcc'))
    args = parser.parse_args()
    if not args.cc:
        parser.error('pass --cc with a C compiler')
    out = ROOT / 'build' / 'test' / 'nvn2-shader-control'
    out.mkdir(parents=True, exist_ok=True)
    src = out / 'control.c'
    src.write_text(source())
    exe = out / ('control.exe' if os.name == 'nt' else 'control')
    subprocess.run([args.cc, '-O2', '-Wall', '-Wextra', '-I', str(ROOT / 'runtime'), str(src), '-o', str(exe)], check=True)
    return subprocess.run([str(exe)]).returncode


if __name__ == '__main__':
    raise SystemExit(main())
