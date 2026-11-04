// Original CPU-only subgroup admission checks, appended to the existing spies.
static Pipe *pipe_miss(const VkPipeState *,const GKey *,uint64_t,int,int *);
static PlatCond work_cv=PLAT_COND_INIT;
static unsigned layout_calls;
static Layout *layout_get(Bindings *b) { (void)b;layout_calls++;return NULL; }
static Layout *layout_pair(Bindings *b,Bindings *f) { (void)b;(void)f;layout_calls++;return NULL; }
static int workers_start(void) { abort(); }
static Pipe rejected;
static Pipe *pipe_add(const void *key,size_t size,uint64_t h,VkPipeline p,const Layout *l,
                      const Bindings *b,VkPipelineBindPoint point,uint32_t inputs) {
    (void)key;(void)size;(void)h;(void)b;(void)point;(void)inputs;
    CHECK(!p && !l); return &rejected;
}
void plat_cond_signal(PlatCond *c) { (void)c;abort(); }
static void subgroup_properties(uint32_t stages,uint32_t operations) {
#ifdef HAS_SUBGROUP_GUARD
    P.subgroup_stages=stages; P.subgroup_operations=operations;
#else
    (void)stages;(void)operations;
#endif
}
static void rejection(VkPipeState *s) {
    GKey k;gkey_fill(s,&k);int queued=-1;
    unsigned calls=pipeline_calls,layouts_before=layout_calls;
    CHECK(geometry_validate(s,&fixture_context.props.limits));
    CHECK(pipe_miss(s,&k,0,0,&queued)==&rejected);
    CHECK(!queued && pipeline_calls==calls && layout_calls==layouts_before);
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    output_directory=argv[1];capabilities();
    GfxShader vs,gs,fs;
    load_shader(&vs,argv[1],"vs",GFX_STAGE_VS);load_shader(&fs,argv[1],"ps",GFX_STAGE_FS);
    VkPipeState s={.vs=&vs,.fs=&fs,.prim=NVN_DRAW_PRIMITIVE_POINTS,.samples=1};
    s.poly.mode=NVN_POLYGON_MODE_FILL;
    const uint32_t all=VK_SUBGROUP_FEATURE_BASIC_BIT|VK_SUBGROUP_FEATURE_VOTE_BIT|
        VK_SUBGROUP_FEATURE_BALLOT_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_BIT|VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
    const char *positive[]={"plain","lane","mask","both","declaration","vote","ballot","shuffle","relative"};
    for(unsigned i=0;i<sizeof positive/sizeof *positive;i++) {
        load_shader(&gs,argv[1],positive[i],GFX_STAGE_GS);s.gs=&gs;
        subgroup_properties(VK_SHADER_STAGE_GEOMETRY_BIT,all);
        CHECK(!geometry_validate(&s,&fixture_context.props.limits));
        GeometryInterface gi;CHECK(geometry_interface(&gs,1,1,&gi));
        CHECK(!gi.mask && gi.builtin_mask==1 && gi.components==4);
        if(i<5) {
            subgroup_properties(0,all);
            if(i>=1 && i<=3) {
                // Conservative system-input subset: global operation properties
                // and an available size pin do not establish GS input support.
                P.sg32_stages=VK_SHADER_STAGE_GEOMETRY_BIT;rejection(&s);P.sg32_stages=0;
            } else CHECK(!geometry_validate(&s,&fixture_context.props.limits));
        } else {
            subgroup_properties(0,all);P.sg32_stages=VK_SHADER_STAGE_GEOMETRY_BIT;
            rejection(&s);P.sg32_stages=0;
            static const uint32_t required[]={VK_SUBGROUP_FEATURE_VOTE_BIT,VK_SUBGROUP_FEATURE_BALLOT_BIT,
                VK_SUBGROUP_FEATURE_SHUFFLE_BIT,VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT};
            subgroup_properties(VK_SHADER_STAGE_GEOMETRY_BIT,all&~required[i-5]);rejection(&s);
        }
        if(i) {
            subgroup_properties(VK_SHADER_STAGE_GEOMETRY_BIT,0);rejection(&s);
        }
        free(gs.spirv);
    }
    load_shader(&gs,argv[1],"ordinary_extension",GFX_STAGE_GS);s.gs=&gs;
    subgroup_properties(0,0);CHECK(!geometry_validate(&s,&fixture_context.props.limits));free(gs.spirv);
    const char *negative[]={"signed","float","array","mask_scalar","output","duplicate","unknown",
        "flat","location","component","invariant","unsupported","scope","extension","missing_cap",
        "uint16","uint64","short_mask"};
    for(unsigned i=0;i<sizeof negative/sizeof *negative;i++) {
        load_shader(&gs,argv[1],negative[i],GFX_STAGE_GS);s.gs=&gs;
        subgroup_properties(VK_SHADER_STAGE_GEOMETRY_BIT,all|VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
        rejection(&s);free(gs.spirv);
    }
    for(unsigned i=0;i<sizeof extension_controls/sizeof *extension_controls;i++) {
        load_shader(&gs,argv[1],extension_controls[i],GFX_STAGE_GS);s.gs=&gs;
        subgroup_properties(VK_SHADER_STAGE_GEOMETRY_BIT,all);
        rejection(&s);free(gs.spirv);
    }
    CHECK(!made && !pipeline_calls);
    printf("%u checks, %u failures (actual GS admission, no loader)\n",checks,failures);
    free(vs.spirv);free(fs.spirv);return failures?1:0;
}
