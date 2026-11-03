// Original CPU-only fixture. The Python runner inserts the named production functions without body edits.
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include "nvn_api.h"
#include "nvn_tex.h"
#include "cfg.h"
#include "platform/platform.h"

#ifndef TEST_GAME_VERSION
#define TEST_GAME_VERSION 0
#endif
const int rt_game_version = TEST_GAME_VERSION;
unsigned rt_trace;
static int checks, failures;
#define REQUIRE(c, why) do { if (!(c)) { fprintf(stderr, "fixture: %s\n", why); ExitProcess(2); } } while (0)
#define CHECK(c, why) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", why); } } while (0)
#define perf_count(...) ((void)0)
#define perf_begin() 0ull
#define perf_end(...) ((void)0)
static SRWLOCK texture_lock = SRWLOCK_INIT;
static PlatMutex owned_lock = PLAT_MUTEX_INIT, rl = PLAT_MUTEX_INIT;
static PlatMutex latch = PLAT_MUTEX_INIT;
static PlatCond changed = PLAT_COND_INIT;
static int image_entered, readback_release, image_calls, is_before, is_race;
static int reader_returned, allow_cpu, cpu_done, wait_entered, wait_release, returned_before_copy;
static uint64_t done = 4;
static const uint64_t producer_fence = 5;
static HANDLE replay_worker;
static int small_pending, unknown_config;
static volatile int32_t n_kept;
enum { DR_GUEST_READ, DR_N };
static uint64_t dr_calls[DR_N], dr_tcalls[DR_N], dr_waits[DR_N], dr_twaits[DR_N], dr_ns[DR_N], dr_tns[DR_N];

typedef struct Tex {
    struct { uint64_t addr; unsigned levels, layers; int mirrored, gpu_owned; } pub;
    tex_t t;
    uint8_t gpu_dirty;
    uint32_t gpu_gen;
    uint64_t gpu_seq, gpu_fence, wr_fence;
    uint8_t *snap;
} Tex;
static Tex *owned_array[1], **owned = owned_array;
static int nowned = 1;
static uint64_t max_owned;
/* REGION_TYPE */
/* REPLAY_TYPES */
/* GENERATION_GLOBALS */

// Real Windows locks and write-watch. No successful Vulkan calls are stubbed or linked.
_Static_assert(sizeof(PlatMutex) == sizeof(SRWLOCK), "mutex storage");
_Static_assert(sizeof(PlatCond) == sizeof(CONDITION_VARIABLE), "condition storage");
void plat_mutex_lock(PlatMutex *m) { AcquireSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_unlock(PlatMutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_lock_shared(PlatMutex *m) { AcquireSRWLockShared((PSRWLOCK)m); }
void plat_mutex_unlock_shared(PlatMutex *m) { ReleaseSRWLockShared((PSRWLOCK)m); }
void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    REQUIRE(SleepConditionVariableSRW((PCONDITION_VARIABLE)c, (PSRWLOCK)m, INFINITE, 0), "condition wait");
}
void plat_cond_broadcast(PlatCond *c) { WakeAllConditionVariable((PCONDITION_VARIABLE)c); }
static void lock(void) { AcquireSRWLockExclusive(&texture_lock); }
static void unlock(void) { ReleaseSRWLockExclusive(&texture_lock); }
void rt_fatal(const char *fmt, ...) {
    va_list a; va_start(a, fmt); vfprintf(stderr, fmt, a); va_end(a); fputc('\n', stderr); ExitProcess(70);
}
void rt_log(const char *fmt, ...) { if (strstr(fmt, "unknown key")) unknown_config++; }
void nvn_called(uint8_t *seen, const char *name) { (void)seen; (void)name; }
int kern_core_away(void) { return 1; }
void kern_core_back(int *away) { *away = 0; }
// Real cfg.c reads only the runner's owned ini. Save/default-location paths must never run in this fixture.
uint8_t *plat_file_read(const char *path, uint64_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    REQUIRE(!fseek(file, 0, SEEK_END), "seek owned ini");
    long length = ftell(file);
    REQUIRE(length >= 0 && length < (1 << 20) && !fseek(file, 0, SEEK_SET), "owned ini size");
    uint8_t *bytes = malloc((size_t)length + 1);
    REQUIRE(bytes && fread(bytes, 1, (size_t)length, file) == (size_t)length, "read owned ini");
    fclose(file); bytes[length] = 0; *size = (uint64_t)length;
    return bytes;
}
int plat_file_exists(const char *path) { (void)path; rt_fatal("unexpected cfg save"); return 0; }
int64_t plat_file_size(const char *path) { (void)path; rt_fatal("unexpected cfg save"); return 0; }
int plat_file_write_atomic(const char *path, const void *data, size_t n) {
    (void)path; (void)data; (void)n; rt_fatal("unexpected cfg save"); return 0;
}
int plat_exe_path(char *out, size_t cap) { (void)out; (void)cap; rt_fatal("unexpected default ini lookup"); return 0; }
uint64_t kern_now_ns(void) { return 0; }
static void log_limited(volatile int32_t *n, const char *fmt, ...) { (void)n; (void)fmt; }
static void flush_note(int kind, const pool_t *p, uint64_t size) { (void)kind; (void)p; (void)size; abort(); }
static uint32_t mip(uint32_t size, unsigned level) { size >>= level; return size ? size : 1; }
int mem_frozen(uint64_t addr) { (void)addr; return 0; }
static uint64_t ww_query(uint64_t addr, uint64_t size) {
    ULONG_PTR n = sizeof gen_pages / sizeof *gen_pages;
    DWORD granularity;
    UINT status = GetWriteWatch(WRITE_WATCH_FLAG_RESET, (void *)(uintptr_t)addr, (SIZE_T)size,
                               (PVOID *)gen_pages, &n, &granularity);
    REQUIRE(!status, "GetWriteWatch");
    return (uint64_t)n;
}
int nvn_texels_box(const nvn_texdesc *, unsigned, unsigned, uint32_t, uint32_t, uint32_t, uint32_t,
                   uint32_t, void *, void *, size_t, int);
static __attribute__((unused)) uint64_t replay_enqueue(ReplayItem *item);
static void vk_be_cpu_synced(uint64_t fence);
static void vk_be_guest_written(uint64_t addr, uint64_t size);
static void vk_be_guest_read(uint64_t addr, uint64_t size);
static uint64_t backend_done(void) { return __atomic_load_n(&done, __ATOMIC_ACQUIRE); }
// Synthetic producer fence, controlled independently from the image download by the main thread.
static int backend_wait(uint64_t value, uint64_t timeout) {
    REQUIRE(value == producer_fence && timeout == UINT64_MAX, "wait only for the actual writer fence");
    plat_mutex_lock(&latch);
    wait_entered = 1;
    plat_cond_broadcast(&changed);
    while (!wait_release) plat_cond_wait(&changed, &latch);
    plat_mutex_unlock(&latch);
    REQUIRE(backend_done() >= value, "released fence is complete");
    return 1;
}
const GfxBackend gfx_vulkan_backend = {
    .guest_written = vk_be_guest_written, .guest_read = vk_be_guest_read, .fence_done = backend_done,
    .fence_wait = backend_wait, .cpu_synced = vk_be_cpu_synced,
};
const GfxBackend *gfx_backend(void) { return &gfx_vulkan_backend; }
void vk_sync_start(void) {}
void vk_present_drain(void) {}
uint64_t vk_sync_fence_done(void) { return backend_done(); }
int vk_sync_fence_wait(uint64_t f, uint64_t timeout) { return backend_wait(f, timeout); }
void vk_texture_write_back_small(uint64_t f) { (void)f; rt_fatal("large-only fixture entered small writeback"); }

// Explicit GPU-readback adapter. Its entry follows the real storage_rewritten generation check;
// its output is stale image data, separate from the write-watched guest allocation.
static int image_texels(Tex *e, const Region *g, const NVNcopyRegion *r, uint8_t *mem, int write) {
    (void)e; (void)r;
    REQUIRE(!write && !is_before, "only the intended image download reaches the adapter");
    plat_mutex_lock(&latch);
    image_calls++;
    image_entered = 1;
    plat_cond_broadcast(&changed);
    while (!readback_release) plat_cond_wait(&changed, &latch);
    plat_mutex_unlock(&latch);
    memset(mem, 0x22, g->img * (size_t)g->nz);
    return 1;
}

/* PRODUCTION_FUNCTIONS */

// Explicit one-item replay scheduling adapter. The item runs the production writeback entry;
// it signals no GPU timeline value, matching RI_WRITE_BACK.
static ReplayItem job;
static DWORD WINAPI execute_writeback(void *unused) {
    (void)unused;
    REQUIRE(job.at && job.queued == 128, "actual read_nowait queued the requested range");
    vk_present_drain();
    vk_texture_write_back(job.at, job.queued);
    return 0;
}
static uint64_t replay_enqueue(ReplayItem *item) {
    REQUIRE(!replay_worker, "one queued background item");
    job = *item;
    replay_worker = CreateThread(NULL, 0, execute_writeback, NULL, 0, NULL);
    REQUIRE(replay_worker, "start replay adapter");
    return 0;
}
static void flush(buf_t *buffer) {
    Cpu cpu = {0};
    cpu.x[0] = (uintptr_t)buffer;
    cpu.x[2] = buffer->size;
    nvnp_nvnBufferFlushMappedRange(&cpu);
}
typedef struct { pool_t *pool; buf_t *buffer; } Reader;
static DWORD WINAPI read_then_reuse(void *arg) {
    Reader *reader = arg;
    Cpu cpu = {0};
    cpu.x[0] = (uintptr_t)reader->pool;
    cpu.x[2] = 128;
    nvnp_nvnMemoryPoolInvalidateMappedRange(&cpu);
    plat_mutex_lock(&latch);
    reader_returned = 1;
    returned_before_copy = !is_before && !readback_release;
    plat_cond_broadcast(&changed);
    while (!allow_cpu) plat_cond_wait(&changed, &latch);
    plat_mutex_unlock(&latch);
    sync_t sync = {.fenced = 1, .fence = producer_fence};
    cpu = (Cpu){0}; cpu.x[0] = (uintptr_t)&sync; cpu.x[1] = UINT64_MAX;
    nvnp_nvnSyncWait(&cpu);
    REQUIRE(cpu.x[0] == NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED, "actual SyncWait observes original completion");
    if (is_race) {
        // This is the SAME invalidating caller, after invalidate returned and its producer fence completed.
        memset(reader->pool->mem, 0x77, (size_t)reader->pool->size);
        flush(reader->buffer);
    }
    plat_mutex_lock(&latch);
    cpu_done = 1;
    plat_cond_broadcast(&changed);
    plat_mutex_unlock(&latch);
    return 0;
}
static int all_bytes(const uint8_t *bytes, uint64_t size, uint8_t expected) {
    for (uint64_t i = 0; i < size; i++) if (bytes[i] != expected) return 0;
    return 1;
}
static void join(HANDLE thread) {
    REQUIRE(WaitForSingleObject(thread, 10000) == WAIT_OBJECT_0, "worker finished");
    CloseHandle(thread);
}

int main(int argc, char **argv) {
    REQUIRE(argc == 2 || argc == 3, "select race, before or stable, with an optional owned ini");
    if (argc == 3) REQUIRE(cfg_load(argv[2]), "read the fixture's own settings file");
    const CfgMeta *meta = cfg_meta("NX2_READ_NOWAIT");
    CHECK(meta && !strcmp(meta->def, "0") && !meta->live, "read-without-wait is registered as default-off/restart-only");
    CHECK(!unknown_config, "fixture ini setting is known");
    is_before = !strcmp(argv[1], "before");
    is_race = !strcmp(argv[1], "race");
    REQUIRE(is_before || is_race || !strcmp(argv[1], "stable"), "recognized scenario");
    const char *explicit_nowait = cfg_get("TOTK_NX2_READ_NOWAIT");
    int expect_deferred = !rt_game_version && gfx_replay_async() && explicit_nowait &&
        *explicit_nowait && strcmp(explicit_nowait, "0");
    nvn_texdesc descriptor = {.target = NVN_TEXTURE_TARGET_2D, .format = NVN_FORMAT_RGBA8,
        .flags = NVN_TEXTURE_FLAGS_LINEAR, .width = 256, .height = 256, .depth = 1, .levels = 1, .stride = 1024};
    nvn_texlayout layout;
    nvn_tex_layout(&descriptor, &layout);
    REQUIRE(layout.size > (64u << 10), "target exceeds explicit CPU-sync small writeback limit");
    uint8_t *storage = VirtualAlloc((void *)0x2000000000ull, (SIZE_T)layout.size,
        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE);
    REQUIRE(storage, "private write-watched storage below guest address limit");
    memset(storage, 0x11, (size_t)layout.size);
    pool_t pool = {.magic = NVN_POOL_MAGIC, .mem = storage, .size = layout.size,
                   .flags = NVN_MEMORY_POOL_FLAGS_CPU_CACHED};
    buf_t buffer = {.magic = MAGIC('B', 'U', 'F', 0), .pool = &pool, .off = 0, .size = layout.size};
    Tex texture = {.pub = {.addr = (uintptr_t)storage, .levels = 1, .layers = 1, .mirrored = 1, .gpu_owned = 1},
        .t = {.d = descriptor, .L = layout, .mem = storage, .pool = &pool},
        .gpu_dirty = 1, .gpu_seq = 1, .gpu_fence = producer_fence, .wr_fence = producer_fence};
    owned[0] = &texture;
    max_owned = layout.size;
    vk_mem_changed((uintptr_t)storage, layout.size, 0, &texture.gpu_gen);
    if (is_before) { memset(storage, 0x77, (size_t)layout.size); flush(&buffer); }
    Reader reader = {&pool, &buffer};
    HANDLE caller = CreateThread(NULL, 0, read_then_reuse, &reader, 0, NULL);
    REQUIRE(caller, "start invalidating caller");

    plat_mutex_lock(&latch);
    while (!reader_returned && !wait_entered && !image_entered) plat_cond_wait(&changed, &latch);
    int deferred = reader_returned;
    __atomic_store_n(&done, producer_fence, __ATOMIC_RELEASE);
    wait_release = 1;
    plat_cond_broadcast(&changed);
    if (!is_before) {
        while (!image_entered) plat_cond_wait(&changed, &latch);
        if (!deferred) {
            CHECK(!reader_returned, "synchronous invalidate remains inside its readback");
            readback_release = 1;
            plat_cond_broadcast(&changed);
        }
    }
    while (!reader_returned) plat_cond_wait(&changed, &latch);
    allow_cpu = 1;
    plat_cond_broadcast(&changed);
    while (!cpu_done) plat_cond_wait(&changed, &latch);
    // A deferred readback is released only after the same caller's post-fence CPU write and flush returned.
    readback_release = 1;
    plat_cond_broadcast(&changed);
    plat_mutex_unlock(&latch);
    join(caller);
    if (replay_worker) join(replay_worker);
    CHECK(!!replay_worker == !!expect_deferred, "only the explicit async opt-in queues a background guest-memory commit");
    CHECK(returned_before_copy == (!is_before && !!expect_deferred), "default/off invalidate waits for its readback before returning");
    uint8_t expected = is_before || is_race ? 0x77 : 0x22;
    CHECK(all_bytes(storage, layout.size, expected), "all guest bytes preserve the newest logical writer");
    CHECK(image_calls == (is_before ? 0 : 1), "before-write and no-write controls use the real generation check");
    printf("%s: %d checks, %d failures; target=%llu bytes, final=0x%02x, invalidate returned before copy=%d\n",
           argv[1], checks, failures, (unsigned long long)layout.size, storage[0], returned_before_copy);
    VirtualFree(storage, 0, MEM_RELEASE);
    return failures ? 1 : 0;
}
