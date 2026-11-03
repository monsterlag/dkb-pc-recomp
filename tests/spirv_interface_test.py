"""Exercise production SPIR-V interface reflection and vertex-output unions with owned declarations.

No translator, Vulkan loader, or GPU is used. Cache tests use platform I/O/thread adapters and an identity
name-hash adapter; reflection, mask generation, cache headers/loaders and vertex-output unions are production code.
Usage: python tests/spirv_interface_test.py [--cc clang] [--source-root PATH]
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def function(source, declaration):
    start = source.index(declaration)
    brace = source.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def source(root):
    shader = (root / 'runtime/gfx_shader.c').read_text()
    pipeline = (root / 'runtime/vk_pipeline.c').read_text()
    enum_start = shader.index('enum { OP_ENTRY_POINT')
    opcodes = shader[enum_start:shader.index('};', enum_start) + 2]
    prelude = r'''
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
enum { GFX_TYPE_FLOAT, GFX_TYPE_SINT, GFX_TYPE_UINT };
enum { GFX_STAGE_VS, GFX_STAGE_FS=4 };
typedef struct { uint8_t type, comps, packed; } GfxShaderIo;
typedef struct {
    const uint32_t *spirv; uint64_t words; uint32_t local[3], in_mask, out_mask;
    GfxShaderIo in[32]; uint32_t in_guard; GfxShaderIo out[32]; uint32_t out_guard;
    int stage; const char *reflection;
} GfxShader;
typedef struct { const GfxShader *vs, *fs; } VkPipeState;
static struct { int vs_outs_all; } P;
enum { VS_OUTS_CAP = 64 };
static struct { int lock; const GfxShader *vs[VS_OUTS_CAP]; uint32_t mask[VS_OUTS_CAP];
                GfxShaderIo io[VS_OUTS_CAP][32]; uint32_t n, full; } VO;
static void plat_mutex_lock(int *lock) { (void)lock; }
static void plat_mutex_unlock(int *lock) { (void)lock; }
#define LOG_ONCE(...) ((void)0)
'''
    extracted = [opcodes, function(shader, 'static void reflect_spirv('),
                 function(shader, 'uint32_t gfx_shader_inputs('),
                 function(shader, 'uint32_t gfx_shader_outputs('),
                 function(pipeline, 'static uint32_t vs_outs('),
                 function(pipeline, 'static uint32_t vs_union_missing(')]
    use_end = shader.index('} UseMask;')
    use_types = shader[shader.rfind('typedef struct {', 0, use_end):shader.index('struct GfxShader {', use_end)]
    use_globals = shader[shader.index('#define USE_FILE_BUCKETS'):shader.index('static UseFile **use_file_slot(')]
    use_defines = '\n'.join(line for line in shader.splitlines()
                            if line.startswith(('#define USE_MAGIC ', '#define USE_VERSION ',
                                                '#define BUCKETS ', '#define PATH_CAP ')))
    cache_adapters = r'''
static char cache_dir[PATH_CAP];
static int cache_lock;
static UseEnt *use_buckets[BUCKETS];
enum { PACK_USE=3, PLAT_PRIO_BELOW_NORMAL };
typedef struct { uint8_t name[32]; uint32_t kind,size; uint64_t offset; } PackEnt;
static PackEnt pack_entries[2], *pack_ix=pack_entries;
static uint32_t pack_n, pack_data[10];
typedef struct { int n,bad; } UsePreload;
static void plat_mutex_lock_shared(int *lock) { (void)lock; }
static void plat_mutex_unlock_shared(int *lock) { (void)lock; }
static int32_t plat_cas32(volatile int32_t *p,int32_t value,int32_t expected) {
    int32_t old=*p; if (old==expected) *p=value; return old;
}
static int32_t plat_xchg32(volatile int32_t *p,int32_t value) { int32_t old=*p; *p=value; return old; }
static void plat_thread_priority(int priority) { (void)priority; }
static uint64_t plat_now_ns(void) { return 0; }
static void rt_log(const char *format,...) { (void)format; }
static void use_name(const uint8_t hash[32],uint8_t name[32]) { memcpy(name,hash,32); }
static uint8_t *read_file(const char *path,uint64_t *size) {
    FILE *f=fopen(path,"rb"); if (!f) return NULL;
    if (fseek(f,0,SEEK_END)) { fclose(f); return NULL; }
    long n=ftell(f); rewind(f);
    uint8_t *data=n>0?malloc((size_t)n+1):NULL;
    if (!data || fread(data,1,(size_t)n,f)!=(size_t)n) { free(data); fclose(f); return NULL; }
    fclose(f); data[n]=0; *size=(uint64_t)n; return data;
}
static int plat_file_write_atomic(const char *path,const void *data,size_t n) {
    FILE *f=fopen(path,"wb"); if (!f) return 0;
    int ok=fwrite(data,1,n,f)==n; return fclose(f)==0 && ok;
}
static int pack_read_at(uint64_t offset,void *out,uint32_t n) {
    if (offset>sizeof pack_data || n>sizeof pack_data-offset) return 0;
    memcpy(out,(uint8_t *)pack_data+offset,n); return 1;
}
static int plat_file_list(const char *path,const char *suffix,int (*cb)(const char *,void *),void *ctx) {
    (void)path; (void)suffix; (void)cb; (void)ctx; return 0; // Loose preload is exercised directly.
}
'''
    cache_functions = [function(shader, declaration) for declaration in (
        'static const char *json_key(', 'static uint32_t json_slot_mask(', 'static UseMask use_of(',
        'static int pack_cmp(', 'static const PackEnt *pack_find(', 'static uint8_t *pack_read(',
        'static void key_hex(', 'static void use_path_of(', 'static void use_path(',
        'static UseFile **use_file_slot(', 'static int hex_byte(', 'static int use_preload_file(',
        'static void use_preload(', 'static UseEnt **use_slot(', 'static void use_put(',
        'static int use_find(', 'static void use_store(')]
    tests = r'''
typedef struct { uint32_t w[1024]; unsigned n; } Module;
static void emit(Module *m, unsigned op, unsigned count, ...) {
    if (m->n + count + 1 > sizeof m->w / sizeof *m->w) abort();
    m->w[m->n++] = (count + 1) << 16 | op;
    va_list args; va_start(args, count);
    for (unsigned i = 0; i < count; i++) m->w[m->n++] = va_arg(args, unsigned);
    va_end(args);
}
static Module begin(void) {
    Module m = {.w={0x07230203, 0x00010300, 0, 128, 0}, .n=5};
    return m;
}
static GfxShader reflect(Module *m) {
    GfxShader s = {.spirv=m->w, .words=m->n, .in_guard=0x12345678, .out_guard=0x87654321};
    reflect_spirv(&s); return s;
}
static int guarded_constant(void) {
    size_t page;
    uint8_t *memory;
#ifdef _WIN32
    SYSTEM_INFO si; GetSystemInfo(&si); page=si.dwPageSize;
    memory=VirtualAlloc(NULL,page*2,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    DWORD old;
    if (!memory || !VirtualProtect(memory+page,page,PAGE_NOACCESS,&old)) return 0;
#else
    page=(size_t)sysconf(_SC_PAGESIZE);
    memory=mmap(NULL,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (memory==MAP_FAILED || mprotect(memory+page,page,PROT_NONE)) return 0;
#endif
    Module m=begin(); emit(&m,21,3,1u,64u,0u); emit(&m,43,3,1u,2u,3u);
    uint32_t *words=(uint32_t *)(memory+page-m.n*4);
    memcpy(words,m.w,m.n*4);
    GfxShader s={.spirv=words,.words=m.n}; reflect_spirv(&s);
    int ok=!s.in_mask && !s.out_mask && s.local[0]==1;
#ifdef _WIN32
    VirtualFree(memory,0,MEM_RELEASE);
#else
    munmap(memory,page*2);
#endif
    return ok;
}
// Owned declaration fragments isolate reflection; they are not executable shader programs.
static Module array(unsigned location, unsigned count, unsigned component, unsigned sc) {
    Module m=begin();
    emit(&m,71,3,7u,30u,location);
    if (component!=UINT32_MAX) emit(&m,71,3,7u,31u,component);
    emit(&m,22,2,1u,32u); emit(&m,23,3,2u,1u,4u);
    emit(&m,21,3,3u,32u,0u); emit(&m,43,3,3u,4u,count);
    emit(&m,28,3,5u,2u,4u); emit(&m,32,3,6u,sc,5u); emit(&m,59,3,6u,7u,sc);
    return m;
}
static uint32_t missing(const GfxShader *vs,const GfxShader *fs,GfxShaderIo io[32]) {
    memset(&VO,0,sizeof VO); VkPipeState state={vs,fs}; return vs_union_missing(&state,io);
}
static int checks, failures;
#define CHECK(c, text) do { checks++; if (!(c)) { failures++; fprintf(stderr,"FAIL: %s (line %d)\n",text,__LINE__); } } while (0)
static void check_io(const GfxShader *s,unsigned first,unsigned count,unsigned type,unsigned comps,unsigned packed,int output) {
    const GfxShaderIo *io=output?s->out:s->in;
    for (unsigned i=first;i<first+count;i++) {
        CHECK(io[i].type==type && io[i].comps==comps && io[i].packed==packed,"per-slot type/components/Component");
    }
    CHECK(s->in_guard==0x12345678 && s->out_guard==0x87654321,"reflection array bounds");
}
static void cache_clear(void) {
    for (unsigned i=0;i<BUCKETS;i++) while (use_buckets[i]) {
        UseEnt *e=use_buckets[i]; use_buckets[i]=e->next; free(e);
    }
    for (unsigned i=0;i<USE_FILE_BUCKETS;i++) while (use_files[i]) {
        UseFile *e=use_files[i]; use_files[i]=e->next; free(e);
    }
    use_files_ready=0; pack_n=0;
}
static void use_fixture(const uint8_t hash[32],uint32_t version,const UseMask *mask,char name[69]) {
    char path[PATH_CAP+80],hex[65]; use_path(hash,path); key_hex(hash,hex);
    snprintf(name,69,"%s.use",hex);
    uint32_t header[5]={USE_MAGIC,version,mask->slots,mask->attrs,mask->fouts};
    CHECK(plat_file_write_atomic(path,header,sizeof header),"write owned use-cache fixture");
}
static void cache_checks(const char *directory) {
    snprintf(cache_dir,sizeof cache_dir,"%s",directory);
    Module vm=array(0,16,UINT32_MAX,1),fm=array(0,16,UINT32_MAX,3);
    GfxShader vs=reflect(&vm),fs=reflect(&fm);
    vs.stage=GFX_STAGE_VS; fs.stage=GFX_STAGE_FS;
    vs.reflection=fs.reflection="{\"tex_queried\":[],\"img_queried\":[]}";
    UseMask vertex=use_of(&vs),fragment=use_of(&fs),old={0,1,1},got={0};
    CHECK(vertex.attrs==0xffff && !vertex.fouts,"production use_of retains all vertex array inputs");
    CHECK(fragment.fouts==0xff && !fragment.attrs,"production use_of retains fragment array outputs through target7");
    uint8_t hash[32]={1}; char name[69];
    use_fixture(hash,1,&old,name);
    CHECK(!use_find(hash,&got),"direct loose loader rejects stale V1 mask1");
    use_store(hash,&vertex);
    CHECK(use_find(hash,&got) && got.attrs==0xffff,"recomputed mask replaces remembered V1 miss");
    char path[PATH_CAP+80]; use_path(hash,path); uint64_t size=0;
    uint32_t *written=(uint32_t *)read_file(path,&size);
    CHECK(written && size==20 && written[1]==2 && written[3]==0xffff,"use_store writes V2 without changing20-byte layout");
    free(written); cache_clear();
    CHECK(use_find(hash,&got) && got.attrs==0xffff,"direct loose loader accepts persisted recomputed V2 maskffff");

    cache_clear(); hash[0]=2; use_fixture(hash,1,&old,name); UsePreload st={0};
    use_preload_file(name,&st); use_files_ready=1;
    CHECK(st.n==0 && st.bad==1 && !use_find(hash,&got),"loose preload rejects V1 before publishing completed table");
    cache_clear(); hash[0]=3; use_fixture(hash,2,&vertex,name); st=(UsePreload){0};
    use_preload_file(name,&st); use_files_ready=1;
    CHECK(st.n==1 && st.bad==0 && use_find(hash,&got) && got.attrs==0xffff,"loose preload accepts recomputed V2 array mask");

    cache_clear(); hash[0]=4;
    pack_entries[0]=(PackEnt){.kind=PACK_USE,.size=20,.offset=0}; memcpy(pack_entries[0].name,hash,32);
    const uint32_t stale[5]={USE_MAGIC,1,0,1,1}; memcpy(pack_data,stale,20); pack_n=1;
    CHECK(!use_find(hash,&got),"direct packed loader rejects stale V1 mask1");
    cache_clear(); pack_data[1]=2; pack_data[3]=vertex.attrs; pack_data[4]=vertex.fouts; pack_n=1;
    CHECK(use_find(hash,&got) && got.attrs==0xffff,"direct packed loader accepts recomputed V2 maskffff");

    cache_clear(); memcpy(pack_data,stale,20); memcpy(pack_entries[0].name,hash,32);
    hash[0]=5; pack_entries[1]=(PackEnt){.kind=PACK_USE,.size=20,.offset=20}; memcpy(pack_entries[1].name,hash,32);
    const uint32_t current[5]={USE_MAGIC,2,fragment.slots,fragment.attrs,fragment.fouts};
    memcpy((uint8_t *)pack_data+20,current,20); pack_n=2; use_preload(NULL);
    CHECK(use_files_ready && use_find(hash,&got) && got.fouts==0xff,"packed preload accepts recomputed V2 fragment-output mask");
    hash[0]=4; CHECK(!use_find(hash,&got),"packed preload rejects stale V1 before publishing completed table");
    cache_clear();
}
int main(int argc,char **argv) {
    Module fm=array(0,16,UINT32_MAX,1), vm=begin();
    for (unsigned i=0;i<6;i++) emit(&vm,71,3,20u+i,30u,i);
    emit(&vm,22,2,1u,32u); emit(&vm,23,3,2u,1u,4u); emit(&vm,32,3,3u,3u,2u);
    for (unsigned i=0;i<6;i++) emit(&vm,59,3,3u,20u+i,3u);
    GfxShader fs=reflect(&fm), vs=reflect(&vm); GfxShaderIo io[32];
    CHECK(fs.in_mask==0xffff,"fragment vec4[16] occupies locations0..15");
    CHECK(vs.out_mask==0x3f,"vertex outputs0..5 retained");
    CHECK(missing(&vs,&fs,io)==0xffc0,"production output union fills locations6..15");
    check_io(&fs,0,16,GFX_TYPE_FLOAT,4,0,0);
    for (unsigned i=6;i<16;i++) CHECK(io[i].type==GFX_TYPE_FLOAT && io[i].comps==4,"union metadata for missing array slots");

    fm=array(27,16,UINT32_MAX,1); fs=reflect(&fm);
    CHECK(fs.in_mask==0xf8000000u,"array clipped at location31");
    check_io(&fs,27,5,GFX_TYPE_FLOAT,4,0,0);
    fm=array(31,0xffffffffu,UINT32_MAX,3); fs=reflect(&fm);
    CHECK(fs.out_mask==0x80000000u,"large unsigned length bounded before location arithmetic");
    check_io(&fs,31,1,GFX_TYPE_FLOAT,4,0,1);
    fm=array(32,16,UINT32_MAX,1); fs=reflect(&fm);
    CHECK(!fs.in_mask,"base location outside reflection window ignored");
    fm=array(0x7fffffffu,16,UINT32_MAX,1); fs=reflect(&fm);
    CHECK(!fs.in_mask,"large location cannot overflow signed location metadata");

    fm=begin(); emit(&fm,71,3,9u,30u,4u);
    emit(&fm,22,2,1u,32u); emit(&fm,23,3,2u,1u,2u); emit(&fm,21,3,3u,32u,0u);
    emit(&fm,43,3,3u,4u,3u); emit(&fm,28,3,5u,2u,4u);
    emit(&fm,43,3,3u,6u,2u); emit(&fm,28,3,7u,5u,6u);
    emit(&fm,32,3,8u,1u,7u); emit(&fm,59,3,8u,9u,1u); fs=reflect(&fm);
    CHECK(fs.in_mask==0x3f0,"nested vec2[2][3] spans six locations");
    check_io(&fs,4,6,GFX_TYPE_FLOAT,2,0,0);

    // Existing scalar/vector types and Component decoration presence, including Component0.
    for (unsigned type=0;type<3;type++) for (unsigned comps=1;comps<=4;comps++) {
        fm=begin(); emit(&fm,71,3,7u,30u,5u); emit(&fm,71,3,7u,31u,0u);
        if (!type) emit(&fm,22,2,1u,32u); else emit(&fm,21,3,1u,32u,type==GFX_TYPE_SINT);
        unsigned elem=1;
        if (comps>1) { emit(&fm,23,3,2u,1u,comps); elem=2; }
        emit(&fm,32,3,6u,1u,elem); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
        CHECK(fs.in_mask==0x20,"existing scalar/vector location");
        check_io(&fs,5,1,type,comps,1,0);
    }
    fm=begin(); emit(&fm,71,3,7u,30u,5u); emit(&fm,71,3,7u,31u,2u);
    emit(&fm,21,3,1u,32u,1u); emit(&fm,23,3,2u,1u,2u); emit(&fm,21,3,3u,32u,0u);
    emit(&fm,43,3,3u,4u,3u); emit(&fm,28,3,5u,2u,4u); emit(&fm,32,3,6u,1u,5u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
    CHECK(fs.in_mask==0xe0,"Component-decorated array span"); check_io(&fs,5,3,GFX_TYPE_SINT,2,1,0);
    CHECK(!missing(&vs,&fs,io),"packed array slots keep existing union exclusion");

    // Invalid/unsupported lengths never turn into invented one-slot arrays.
    for (unsigned mode=0;mode<6;mode++) {
        fm=begin(); emit(&fm,71,3,7u,30u,2u); emit(&fm,22,2,1u,32u); emit(&fm,23,3,2u,1u,4u);
        if (mode==3) emit(&fm,22,2,3u,32u); else emit(&fm,21,3,3u,32u,mode==1);
        if (mode!=2) emit(&fm,mode==4?50u:43u,3,3u,4u,mode==0?0u:mode==1?0xffffffffu:3u);
        emit(&fm,28,3,5u,2u,mode==5?200u:4u); emit(&fm,32,3,6u,1u,5u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
        CHECK(!fs.in_mask,"zero/negative/missing/float/specialization/out-of-bound array length ignored");
    }
    for (unsigned sign=0;sign<2;sign++) {
        fm=begin(); emit(&fm,71,3,7u,30u,30u); emit(&fm,22,2,1u,32u); emit(&fm,23,3,2u,1u,4u);
        emit(&fm,21,3,3u,64u,sign); emit(&fm,43,4,3u,4u,0u,sign?0x80000000u:1u);
        emit(&fm,28,3,5u,2u,4u); emit(&fm,32,3,6u,1u,5u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
        CHECK(fs.in_mask==(sign?0u:0xc0000000u),"64-bit array length sign and saturation");
    }
    fm=begin(); emit(&fm,71,3,7u,30u,2u); emit(&fm,22,2,1u,32u); emit(&fm,23,3,2u,1u,4u);
    emit(&fm,21,3,3u,64u,1u); emit(&fm,43,4,3u,4u,3u,0u);
    emit(&fm,28,3,5u,2u,4u); emit(&fm,32,3,6u,1u,5u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
    CHECK(fs.in_mask==0x1c,"small positive 64-bit length");
    for (unsigned width=16;width<=64;width*=4) {
        fm=begin(); emit(&fm,71,3,7u,30u,2u); emit(&fm,22,2,1u,width); emit(&fm,23,3,2u,1u,4u);
        emit(&fm,21,3,3u,32u,0u); emit(&fm,43,3,3u,4u,3u);
        emit(&fm,28,3,5u,2u,4u); emit(&fm,32,3,6u,1u,5u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
        CHECK(!fs.in_mask,"array element widths absent from GfxShaderIo remain unsupported");
    }
    fm=begin(); emit(&fm,71,3,7u,30u,2u); emit(&fm,22,2,1u,32u); emit(&fm,23,3,2u,1u,4u);
    emit(&fm,29,2,5u,2u); emit(&fm,32,3,6u,1u,5u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
    CHECK(!fs.in_mask,"runtime array does not invent a fixed span");
    fm=array(29,0xffffffffu,UINT32_MAX,1); // Nested large spans cannot overflow back to zero.
    fm.n-=8; emit(&fm,28,3,8u,5u,4u); emit(&fm,32,3,6u,1u,8u); emit(&fm,59,3,6u,7u,1u); fs=reflect(&fm);
    CHECK(fs.in_mask==0xe0000000u,"nested large array span saturates safely");
    fm=begin(); emit(&fm,16,5,1u,17u,4u,5u,6u); fs=reflect(&fm);
    CHECK(fs.local[0]==4 && fs.local[1]==5 && fs.local[2]==6,"compute LocalSize retained");
    fm=array(0,16,UINT32_MAX,1); fm.w[fm.n++]=7u<<16|71u; fs=reflect(&fm);
    CHECK(fs.in_mask==0xffff,"truncated trailing instruction stops without losing reflected slots");
    CHECK(guarded_constant(),"short 64-bit constant ends at a protected page without an extra-word read");
    if (argc!=2) return 2;
    cache_checks(argv[1]);
    printf("SPIR-V interfaces: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
'''
    return '\n'.join([prelude,*extracted,use_defines,use_types,use_globals,cache_adapters,*cache_functions,tests])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cc',default=os.environ.get('CC','clang'))
    parser.add_argument('--source-root',type=Path,default=ROOT)
    args=parser.parse_args()
    cc=shutil.which(args.cc) or args.cc
    if not Path(cc).is_file():
        parser.error('C compiler unavailable; pass --cc PATH')
    build=ROOT/'build'/'tests'/'spirv-interface'
    build.mkdir(parents=True,exist_ok=True)
    cfile=build/'spirv-interface.c'
    exe=build/('spirv-interface.exe' if os.name=='nt' else 'spirv-interface')
    cfile.write_text(source(args.source_root.resolve()))
    subprocess.run([str(cc),'-std=c11','-O1','-Wall','-Wextra','-Werror',str(cfile),'-o',str(exe)],check=True)
    cache = tempfile.mkdtemp(prefix='use-cache-',dir=build)
    return subprocess.run([str(exe),cache]).returncode


if __name__=='__main__':
    raise SystemExit(main())
