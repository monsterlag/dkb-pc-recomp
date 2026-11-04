// Original CPU fixtures for range planning and preflight validation. The bounded
// writer below is a test consumer of the plan, not a Vulkan/reset implementation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_indirect_reset.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

static unsigned checks, failures;
#define CHECK(c, message) do { checks++; if (!(c)) { failures++; fprintf(stderr,"FAIL line %d: %s\n",__LINE__,message); } } while (0)

static void ranges(void) {
    static const struct {
        const char *name;
        uint64_t address;
        uint32_t size;
        int64_t first, count;
        uint64_t begin;
        uint32_t bytes, records;
    } cases[] = {
        {"whole list",0x1000,32,0,4,0x1000,32,4},
        {"nonzero first",0x1000,64,3,5,0x1018,40,5},
        {"incomplete unused tail",0x1000,23,1,1,0x1008,8,1},
        {"last whole entry of maximum descriptor",0x1000,UINT32_MAX,536870910,1,UINT64_C(0x100000ff0),8,1},
        {"maximum bounded work",0x1000,8388608,0,1048576,0x1000,8388608,1048576},
        {"last representable exclusive end",UINT64_MAX-8,8,0,1,UINT64_MAX-8,8,1},
        {"null list",0,32,0,1,0,0,0},
        {"zero descriptor size",0x1000,0,0,1,0,0,0},
        {"short descriptor",0x1000,7,0,1,0,0,0},
        {"negative first",0x1000,32,-1,1,0,0,0},
        {"most negative first",0x1000,32,INT64_MIN,1,0,0,0},
        {"negative count",0x1000,32,0,-1,0,0,0},
        {"zero count",0x1000,32,0,0,0,0,0},
        {"first at descriptor end",0x1000,32,4,1,0,0,0},
        {"first beyond descriptor end",0x1000,32,5,1,0,0,0},
        {"selected tail exceeds descriptor",0x1000,32,3,2,0,0,0},
        {"work cap exceeded",0x1000,UINT32_MAX,0,1048577,0,0,0},
        {"very large first",0x1000,UINT32_MAX,INT64_MAX,1,0,0,0},
        {"very large count",0x1000,UINT32_MAX,0,INT64_MAX,0,0,0},
        {"first offset wraps",UINT64_MAX-7,32,1,1,0,0,0},
        {"selected end wraps",UINT64_MAX-15,16,1,1,0,0,0}
    };
    for (unsigned i=0;i<sizeof cases/sizeof *cases;i++) {
        struct { uint64_t before; NvnIndirectResetList plan; uint64_t after; } guarded;
        memset(&guarded,0xa5,sizeof guarded);
        uint64_t before=guarded.before,after=guarded.after;
        int ok=nvn_indirect_reset_list(cases[i].address,cases[i].size,cases[i].first,cases[i].count,&guarded.plan);
        CHECK(ok==(cases[i].records!=0),cases[i].name);
        CHECK(guarded.plan.begin==cases[i].begin && guarded.plan.bytes==cases[i].bytes &&
              guarded.plan.count==cases[i].records,"exact range or cleared failure output");
        CHECK(guarded.before==before && guarded.after==after,"plan output canaries");
    }
    CHECK(!nvn_indirect_reset_list(0x1000,8,0,1,NULL),"missing plan output rejected");
}

typedef struct { uint64_t address; uint32_t size; uint8_t *data; } Region;
typedef struct { Region region[4]; unsigned n, queries; } Memory;

static uint8_t *mapping(Memory *memory,uint64_t address,uint32_t bytes) {
    for (unsigned i=0;i<memory->n;i++) {
        Region *r=&memory->region[i];
        if (address>=r->address && address-r->address<=r->size && bytes<=r->size-(address-r->address))
            return r->data+(address-r->address);
    }
    return NULL;
}

static int mapped(void *context,uint64_t address,uint32_t bytes) {
    Memory *memory=context;
    memory->queries++;
    CHECK(bytes==20,"mapping query covers exactly five words");
    return mapping(memory,address,bytes)!=NULL;
}

static int everything_mapped(void *context,uint64_t address,uint32_t bytes) {
    (void)context; (void)address;
    CHECK(bytes==20,"opaque target query size");
    return 1;
}

static int never_mapped(void *context,uint64_t address,uint32_t bytes) {
    (void)address; (void)bytes;
    (*(unsigned *)context)++;
    return 0;
}

// Consumers must validate the complete copied list first. Fake GPU bytes here are
// bounded owned CPU storage, and invalid writes remain safe even in negative controls.
static int apply_checked(const NvnIndirectResetList *plan,const uint64_t *pointers,Memory *memory) {
    if (!nvn_indirect_reset_targets(plan,pointers,mapped,memory)) return 0;
    static const uint32_t words[5]={0,1,0,0,0};
    for (uint32_t i=0;i<plan->count;i++) {
        if (!pointers[i]) continue;
        uint8_t *destination=mapping(memory,pointers[i],sizeof words);
        if (!destination) return -1;
        memcpy(destination,words,sizeof words);
    }
    return 1;
}

static void targets(void) {
    uint32_t data[96],before[96];
    for (unsigned i=0;i<96;i++) data[i]=0xa5c35a7e;
    memcpy(before,data,sizeof data);
    Memory memory={.region={{0x20000000,sizeof data,(uint8_t *)data}},.n=1};
    NvnIndirectResetList plan;
    CHECK(nvn_indirect_reset_list(0x10000000,40,0,5,&plan),"original five-entry fixture");
    const uint64_t pointers[5]={0x20000010,0,0x20000040,0x20000010,0x20000070};
    uint64_t unchanged[5]; memcpy(unchanged,pointers,sizeof pointers);
    CHECK(nvn_indirect_reset_targets(&plan,pointers,mapped,&memory),"null and duplicate targets accepted");
    CHECK(!memcmp(data,before,sizeof data) && !memcmp(pointers,unchanged,sizeof pointers),"preflight changes neither records nor copied pointers");
    CHECK(apply_checked(&plan,pointers,&memory)==1,"validated consumer completes");
    for (unsigned i=0;i<96;i++) {
        uint32_t expected=0xa5c35a7e;
        if ((i>=4 && i<9) || (i>=16 && i<21) || (i>=28 && i<33))
            expected=(i==5 || i==17 || i==29)?1:0;
        CHECK(data[i]==expected,"exact record words and untouched surrounding canaries");
    }

    memcpy(data,before,sizeof data);
    CHECK(nvn_indirect_reset_list(0x10000000,32,0,4,&plan),"late-failure fixture range");
    const uint64_t late_bad[4]={0x20000010,0x20000040,0x20000070,0x50000000};
    CHECK(apply_checked(&plan,late_bad,&memory)==0,"last unmapped target rejects the complete reset");
    CHECK(!memcmp(data,before,sizeof data),"late invalid target cannot leave partial reset writes");

    CHECK(nvn_indirect_reset_list(0x10000000,16,0,2,&plan),"overlapping-target fixture range");
    const uint64_t overlap[2]={0x20000010,0x20000014};
    CHECK(apply_checked(&plan,overlap,&memory)==1,"target-to-target overlap retains sequential write semantics");
    const uint32_t ordered[6]={0,0,1,0,0,0};
    CHECK(!memcmp(data+4,ordered,sizeof ordered) && data[3]==before[3] && data[10]==before[10],"overlapping records preserve original pointer order");

    NvnIndirectResetList selected;
    CHECK(nvn_indirect_reset_list(0x30000000,96,4,2,&selected) && selected.begin==0x30000020 && selected.bytes==16,
          "selected pointer subrange excludes unused descriptor entries");
    const uint64_t outside_selected[2]={0x3000000c,0x30000030};
    CHECK(nvn_indirect_reset_targets(&selected,outside_selected,everything_mapped,NULL),"targets ending at selected begin or starting at selected end are allowed");
    const uint64_t unused_suffix[2]={0x30000000,0x30000040};
    CHECK(nvn_indirect_reset_targets(&selected,unused_suffix,everything_mapped,NULL),"unused descriptor prefix and suffix are not forbidden overlap");
    const uint64_t overlap_list[2]={0x20000010,0x30000010};
    CHECK(!nvn_indirect_reset_targets(&selected,overlap_list,everything_mapped,NULL),"record crossing into selected list is rejected");
    const uint64_t unaligned[2]={0x20000011,0};
    CHECK(!nvn_indirect_reset_targets(&selected,unaligned,everything_mapped,NULL),"unaligned record rejected even if mapped");
    const uint64_t wrapping[2]={UINT64_MAX-15,0};
    CHECK(!nvn_indirect_reset_targets(&selected,wrapping,everything_mapped,NULL),"record end overflow rejected even if mapped");
    const uint64_t high_valid[2]={UINT64_MAX-23,0};
    CHECK(nvn_indirect_reset_targets(&selected,high_valid,everything_mapped,NULL),"representable high record address is not truncated");
    unsigned queries=0;
    const uint64_t nulls[2]={0,0};
    CHECK(nvn_indirect_reset_targets(&selected,nulls,never_mapped,&queries) && !queries,"all-null list needs no target mapping");
    CHECK(!nvn_indirect_reset_targets(NULL,nulls,everything_mapped,NULL),"missing list rejected");
    CHECK(!nvn_indirect_reset_targets(&selected,NULL,everything_mapped,NULL),"missing copied pointers rejected");
    CHECK(!nvn_indirect_reset_targets(&selected,nulls,NULL,NULL),"missing mapping query rejected");
    NvnIndirectResetList malformed=selected; malformed.bytes--;
    CHECK(!nvn_indirect_reset_targets(&malformed,nulls,everything_mapped,NULL),"inconsistent copied-list size rejected");
    malformed=selected; malformed.begin=UINT64_MAX-7;
    CHECK(!nvn_indirect_reset_targets(&malformed,nulls,everything_mapped,NULL),"forged list end overflow rejected");

    Memory split={.region={{0x40000000,16,(uint8_t *)data},{0x40000010,64,(uint8_t *)data+16}},.n=2};
    const uint64_t crossing[2]={0x40000008,0};
    CHECK(!nvn_indirect_reset_targets(&selected,crossing,mapped,&split),"fully covered union of two mappings is not one contiguous record");
    const uint64_t contiguous[2]={0x40000010,0};
    CHECK(nvn_indirect_reset_targets(&selected,contiguous,mapped,&split),"whole record inside one mapping accepted");
}

static void protected_tail(void) {
    size_t page;
    uint8_t *allocation;
#ifdef _WIN32
    SYSTEM_INFO info; GetSystemInfo(&info); page=info.dwPageSize;
    allocation=VirtualAlloc(NULL,page*2,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE);
    DWORD old;
    if (!allocation || !VirtualProtect(allocation+page,page,PAGE_NOACCESS,&old)) abort();
#else
    page=(size_t)sysconf(_SC_PAGESIZE);
    allocation=mmap(NULL,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (allocation==MAP_FAILED || mprotect(allocation+page,page,PROT_NONE)) abort();
#endif
    uint64_t *pointers=(uint64_t *)(allocation+page-3*sizeof(uint64_t));
    pointers[0]=0x20000010; pointers[1]=0; pointers[2]=0x20000040;
    NvnIndirectResetList list;
    CHECK(nvn_indirect_reset_list(0x10000000,24,0,3,&list),"protected-tail list range");
    CHECK(nvn_indirect_reset_targets(&list,pointers,everything_mapped,NULL),"last copied pointer ends at a protected page");
    CHECK(pointers[0]==0x20000010 && !pointers[1] && pointers[2]==0x20000040,"copied pointer tail remains unchanged");
#ifdef _WIN32
    if (!VirtualFree(allocation,0,MEM_RELEASE)) abort();
#else
    if (munmap(allocation,page*2)) abort();
#endif
}

int main(void) {
    ranges(); targets(); protected_tail();
    printf("indirect reset planner: %u checks, %u failures\n",checks,failures);
    return failures!=0;
}
