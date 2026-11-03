// Real replay_main/enqueue/drain, with only host scheduling and GPU work replaced by deterministic latches.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif
#include "nvn_obj.h"
#include "platform/platform.h"

enum { MAIN_THREAD, REPLAY_THREAD, DRAIN_THREAD };
static _Thread_local int role;
static int enabled = 1, stop_worker;
static PlatMutex test_lock = PLAT_MUTEX_INIT;
static PlatCond test_changed = PLAT_COND_INIT;
static struct { int entered, release, done; } gates[3];
static int captured, resume_core_away, initial_wait, blocked_on_later, drain_returned, away_calls, back_calls;
static uint64_t reserved, signalled;
// Only the frame field and profiling hooks cross this queue slice; no CPU/lifter or Vulkan headers are needed.
static struct { uint64_t frames; } gfx_counters;
enum { PERF_DRAIN };
int gfx_replay_async(void);
void perf_replay_thread(void);
uint64_t perf_begin(void);
void perf_end(int kind, uint64_t at);
uint64_t kern_now_ns(void);
int kern_core_away(void);
void kern_core_back(int *away);
void rt_fatal(const char *fmt, ...);

static uint64_t submit_sync(const uint64_t *handles, int n);
static void present_image(const tex_t *t, const int crop[4], uint64_t at, uint64_t queued, uint64_t frame, uint64_t value);
void vk_present_drain(void);
void vk_texture_release(const tex_t *t);
uint64_t vk_sync_reserve(void);

// Generated in ignored build/ by replay_drain_test.py from the selected runtime revision, without rewriting it.
#include "replay_queue.inc"

static int checks, failures;
#define CHECK(c, text) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", text); } } while (0)

void rt_fatal(const char *fmt, ...) {
    va_list args; va_start(args, fmt); vfprintf(stderr, fmt, args); va_end(args);
    fputc('\n', stderr); exit(70);
}
int gfx_replay_async(void) { return enabled; }
void perf_replay_thread(void) {}
uint64_t perf_begin(void) { return 1; }
void perf_end(int kind, uint64_t at) { (void)kind; (void)at; }
uint64_t kern_now_ns(void) { return 2; }
int kern_core_away(void) {
    if (role == DRAIN_THREAD) {
        plat_mutex_lock(&test_lock);
        captured = 1; away_calls++;
        plat_cond_broadcast(&test_changed);
        while (!resume_core_away) plat_cond_wait(&test_changed, &test_lock);
        plat_mutex_unlock(&test_lock);
    }
    return 1;
}
void kern_core_back(int *away) {
    if (role == DRAIN_THREAD) {
        plat_mutex_lock(&test_lock); back_calls++; plat_mutex_unlock(&test_lock);
    }
    *away = 0;
}

#ifdef _WIN32
_Static_assert(sizeof(PlatMutex) == sizeof(SRWLOCK) && sizeof(PlatCond) == sizeof(CONDITION_VARIABLE), "lock storage");
void plat_mutex_lock(PlatMutex *m) { AcquireSRWLockExclusive((PSRWLOCK)m); }
void plat_mutex_unlock(PlatMutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)m); }
void plat_cond_broadcast(PlatCond *c) { WakeAllConditionVariable((PCONDITION_VARIABLE)c); }
typedef HANDLE Thread;
typedef struct { PlatThreadFn fn; void *arg; int role; } Start;
static DWORD WINAPI entry(void *arg) {
    Start s = *(Start *)arg; free(arg); role = s.role; s.fn(s.arg); return 0;
}
static Thread start(PlatThreadFn fn, void *arg, int thread_role) {
    Start *s = malloc(sizeof *s); if (!s) abort(); *s = (Start){fn, arg, thread_role};
    Thread t = CreateThread(NULL, 0, entry, s, 0, NULL); if (!t) abort(); return t;
}
static void join(Thread t) { WaitForSingleObject(t, INFINITE); CloseHandle(t); }
#else
static void *lazy(void *volatile *slot, int cond) {
    void *p = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
    if (p) return p;
    void *fresh = malloc(cond ? sizeof(pthread_cond_t) : sizeof(pthread_mutex_t));
    if (!fresh) abort();
    if (cond) pthread_cond_init(fresh, NULL); else pthread_mutex_init(fresh, NULL);
    if (__atomic_compare_exchange_n(slot, &p, fresh, 0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) return fresh;
    if (cond) pthread_cond_destroy(fresh); else pthread_mutex_destroy(fresh);
    free(fresh); return p;
}
void plat_mutex_lock(PlatMutex *m) { pthread_mutex_lock(lazy(&m->p, 0)); }
void plat_mutex_unlock(PlatMutex *m) { pthread_mutex_unlock(m->p); }
void plat_cond_broadcast(PlatCond *c) { pthread_cond_broadcast(lazy(&c->p, 1)); }
typedef pthread_t Thread;
typedef struct { PlatThreadFn fn; void *arg; int role; } Start;
static void *entry(void *arg) {
    Start s = *(Start *)arg; free(arg); role = s.role; s.fn(s.arg); return NULL;
}
static Thread start(PlatThreadFn fn, void *arg, int thread_role) {
    Start *s = malloc(sizeof *s); if (!s) abort(); *s = (Start){fn, arg, thread_role};
    Thread t; if (pthread_create(&t, NULL, entry, s)) abort(); return t;
}
static void join(Thread t) { pthread_join(t, NULL); }
#endif

static Thread worker;
int plat_thread_spawn(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    (void)name; (void)stack; worker = start(fn, arg, REPLAY_THREAD); return 1;
}
void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    if (c == &rcv && role == REPLAY_THREAD && stop_worker) {
        // Stop the fixture's idle infinite worker after every queued item was processed.
        plat_mutex_unlock(m);
#ifdef _WIN32
        ExitThread(0);
#else
        pthread_exit(NULL);
#endif
    }
    if (c == &rcv && role == DRAIN_THREAD) {
        // rl is held. B was queued before the first sleep. Once rq_n==0 with a busy worker, A has been
        // accounted and the worker has taken B. This observes real worker progress without using PR counters.
        plat_mutex_lock(&test_lock);
        if (initial_wait && gates[1].done && !rq_n && r_busy) blocked_on_later = 1;
        initial_wait = 1;
        plat_cond_broadcast(&test_changed);
        plat_mutex_unlock(&test_lock);
    }
#ifdef _WIN32
    if (!SleepConditionVariableSRW((PCONDITION_VARIABLE)c, (PSRWLOCK)m, INFINITE, 0)) abort();
#else
    pthread_cond_wait(lazy(&c->p, 1), m->p);
#endif
}

static void work(unsigned id) {
    if (id < 1 || id > 2) abort();
    plat_mutex_lock(&test_lock);
    gates[id].entered = 1; plat_cond_broadcast(&test_changed);
    while (!gates[id].release) plat_cond_wait(&test_changed, &test_lock);
    gates[id].done = 1; plat_cond_broadcast(&test_changed);
    plat_mutex_unlock(&test_lock);
}
static uint64_t submit_sync(const uint64_t *handles, int n) {
    if (n != 1) abort();
    work((unsigned)handles[0]); return ++signalled;
}
static void present_image(const tex_t *t, const int crop[4], uint64_t at, uint64_t queued, uint64_t frame, uint64_t value) {
    (void)crop; (void)at; (void)queued; (void)frame;
    work((unsigned)t->serial);
    if (++signalled != value) abort();
}
void vk_present_drain(void) {}
void vk_texture_release(const tex_t *t) { work((unsigned)t->serial); }
uint64_t vk_sync_reserve(void) { return ++reserved; }   // called by the actual enqueue while rl is held

static tex_t *enqueue(int kind, unsigned id) {
    ReplayItem item = {.kind=kind, .submit_no=id};
    tex_t *t = NULL;
    if (kind == RI_SUBMIT) {
        item.h = malloc(sizeof *item.h); if (!item.h) abort();
        item.h[0] = id; item.n = 1;
    } else {
        t = calloc(1, sizeof *t); if (!t) abort(); t->serial = id; item.tex = t;
    }
    uint64_t value = replay_enqueue(&item);
    CHECK((kind == RI_RELEASE) == !value, "release work is enqueued without a GPU timeline value");
    return kind == RI_PRESENT ? t : NULL;   // replay_main owns release objects
}
static void drain(void *arg) {
    (void)arg;
    replay_drain(DR_POOL_FINI);
    plat_mutex_lock(&test_lock); drain_returned = 1; plat_cond_broadcast(&test_changed); plat_mutex_unlock(&test_lock);
}
static void scenario(int earlier, int later) {
    plat_mutex_lock(&test_lock);
    memset(gates, 0, sizeof gates);
    captured = resume_core_away = initial_wait = blocked_on_later = drain_returned = away_calls = back_calls = 0;
    plat_mutex_unlock(&test_lock);
    tex_t *a = enqueue(earlier, 1);
    plat_mutex_lock(&test_lock);
    while (!gates[1].entered) plat_cond_wait(&test_changed, &test_lock);
    plat_mutex_unlock(&test_lock);
    plat_mutex_lock(&rl);
    CHECK(r_busy && !rq_n, "the prefix contains an active item even when the pending ring is empty");
    plat_mutex_unlock(&rl);
    Thread drainer = start(drain, NULL, DRAIN_THREAD);
    plat_mutex_lock(&test_lock);
    while (!captured) plat_cond_wait(&test_changed, &test_lock);
    CHECK(!drain_returned && !gates[1].done, "drain waits for the earlier active item");
    plat_mutex_unlock(&test_lock);
    tex_t *b = enqueue(later, 2);   // queued after the actual drain captured its prefix and dropped rl
    plat_mutex_lock(&test_lock);
    resume_core_away = 1; plat_cond_broadcast(&test_changed);
    while (!initial_wait) plat_cond_wait(&test_changed, &test_lock);
    gates[1].release = 1; plat_cond_broadcast(&test_changed);
    while (!drain_returned && !blocked_on_later) plat_cond_wait(&test_changed, &test_lock);
    while (!gates[2].entered) plat_cond_wait(&test_changed, &test_lock);
    CHECK(drain_returned && !blocked_on_later, "later enqueues cannot extend the captured drain prefix");
    CHECK(!gates[2].done && !gates[2].release, "drain result is observed while the later item is still blocked");
    gates[2].release = 1; plat_cond_broadcast(&test_changed);
    plat_mutex_unlock(&test_lock);
    join(drainer);
    replay_drain(DR_FINISH);   // cleanup only: no later producer can extend this drain
    CHECK(away_calls == 1 && back_calls == 1, "the waiting guest core is released and restored once");
    free(a); free(b);
}
int main(void) {
    // Read the unchanged shared-state members whose producers are outside this extracted queue slice.
    (void)r_presents_queued; (void)game_frames;
    scenario(RI_SUBMIT, RI_RELEASE);
    scenario(RI_RELEASE, RI_SUBMIT);
    scenario(RI_PRESENT, RI_RELEASE);
    scenario(RI_RELEASE, RI_RELEASE);
    enabled = 0;
    replay_drain(DR_FINISH);
    CHECK(!rq_n && !r_busy, "synchronous-mode drain leaves the idle queue alone");
    plat_mutex_lock(&rl); stop_worker = 1; plat_cond_broadcast(&rcv); plat_mutex_unlock(&rl);
    join(worker);
    printf("replay drain: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
