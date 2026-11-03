// Synthetic compute-root state and resource tables; no shader binaries or game data.
#include <stdio.h>
#include "nvn2_c0.h"

#ifdef NVN2_C0_BASELINE
int baseline_pack(uint8_t *, size_t, const Nvn2C0Layout *, const uint32_t [8][64], const p_bind_buf [16], const p_bind_buf [16]);
#define nvn2_c0_pack baseline_pack
#endif

static int checks, failures;
#define CHECK(c, message) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", message); } } while (0)
static uint32_t word(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t address(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static int zero(const uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) if (p[i]) return 0; return 1; }

int main(void) {
    uint8_t bank[NVN2_C0_BYTES], previous[NVN2_C0_BYTES];
    uint32_t roots[8][64];
    p_bind_buf ubo[16] = {{0}}, ssbo[16] = {{0}};
    for (unsigned t = 0; t < 8; t++)
        for (unsigned i = 0; i < 64; i++) roots[t][i] = 0xa0000000u + 0x100u * t + i;
    ubo[0].addr = UINT64_C(0x1122334455667788), ubo[0].size = 0x1234;
    ssbo[0].addr = UINT64_C(0x2233445566778899), ssbo[0].size = 0x3456;
    ssbo[1].addr = UINT64_C(0x33445566778899aa), ssbo[1].size = UINT64_C(0x100004000);

    // Images=7 does not imply 7 rectangles: roots 0 and 2 contain 3 and 5 words, padded independently.
    Nvn2C0Layout l = {.images=7, .ubos=1, .ssbos=2, .textures=3, .roots={3,0,5}};
    CHECK(nvn2_c0_part(&l, 1) == 0x40, "buffer tables follow separately padded root tables");
    CHECK(nvn2_c0_part(&l, 2) == 0x50 && nvn2_c0_part(&l, 3) == 0x70, "uniform/storage table ordering");
    CHECK(nvn2_c0_part(&l, 4) == 0x88 && nvn2_c0_part(&l, 5) == 0xc0, "texture/image table boundaries");
    memset(bank, 0xcc, sizeof bank);
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "pack independent image/root counts");
    CHECK(zero(bank, 16), "driver head remains separate from root data");
    CHECK(!memcmp(bank + 0x10, roots[0], 12), "first root contains the pushed words");
    CHECK(zero(bank + 0x1c, 4), "oversized stored push does not overwrite root padding");
    CHECK(!memcmp(bank + 0x20, roots[2], 20), "empty root 1 does not occupy space before root 2");
    CHECK(zero(bank + 0x34, 12), "second root padding is zero");
    CHECK(address(bank + 0x40) == ubo[0].addr && word(bank + 0x48) == 0x1234, "uniform address/size preserved");
    CHECK(address(bank + 0x50) == ssbo[0].addr && word(bank + 0x58) == 0x3456, "storage address/size preserved");
    CHECK(address(bank + 0x60) == ssbo[1].addr && word(bank + 0x68) == 0x4000, "storage size keeps its 32-bit ABI");
    CHECK(zero(bank + 0x4c, 4) && zero(bank + 0x5c, 4) && zero(bank + 0x6c, 4), "table padding is not stale root data");
    CHECK(zero(bank + 0x70, sizeof bank - 0x70), "unbound handles and unused bank bytes start at zero");

    memcpy(previous, bank, sizeof bank);
    l.images = 1;
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "changing image count still packs");
    CHECK(!memcmp(previous, bank, 0x70), "image count cannot move or change root/buffer data");

    roots[0][0] = 0;
    l.roots[0] = 1;
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "smaller declared root still packs");
    CHECK(zero(bank + 0x10, 16), "a zero push clears its word and stale undeclared words do not leak");
    roots[0][0] = 0x87654321;
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo) && word(bank + 0x10) == 0x87654321,
          "a later push refreshes the declared root word");

    memset(l.roots, 0, sizeof l.roots);
    l.images = 7;
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "no declared roots with nonzero image count");
    CHECK(nvn2_c0_part(&l, 1) == 0x10 && address(bank + 0x10) == ubo[0].addr,
          "undeclared stale pushes cannot replace a buffer table");

    memset(&l, 0, sizeof l);
    l.roots[7] = 5;
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo) && !memcmp(bank + 0x10, roots[7], 20),
          "root table 7 is serialized after empty preceding tables");
    memset(roots, 0, sizeof roots);
    CHECK(nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo) && zero(bank, sizeof bank),
          "unwritten root state is zero without guessing image dimensions");

    memset(bank, 0xcc, sizeof bank);
    memcpy(previous, bank, sizeof bank);
    l.roots[0] = 65;
    CHECK(!nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "reject root count beyond API capacity");
    CHECK(!memcmp(bank, previous, sizeof bank), "rejected layout leaves destination unchanged");
    l.roots[0] = 64;
    CHECK(!nvn2_c0_pack(bank, 16, &l, roots, ssbo, ubo), "reject truncated destination before copying");
    memset(l.roots, 64, sizeof l.roots);
    CHECK(!nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "reject combined block beyond driver bank");
    memset(l.roots, 0, sizeof l.roots);
    l.ssbos = 17;
    CHECK(!nvn2_c0_pack(bank, sizeof bank, &l, roots, ssbo, ubo), "reject driver table beyond bound-slot capacity");
    printf("nvn2 compute c0: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
