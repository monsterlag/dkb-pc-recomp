// Adapters exercise the production clear dispatcher without a Vulkan device. The GPU probe exercises its real
// fixed-function helper separately. These owned cases check routing, scissor, view layers and state invalidation.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vk_exec.h"

VkExecFrame vk_exec_frame;
enum { W_CLEAR_TARGET, W_CLEAR_MASK, W_STENCIL_MASK };
static int failures, checks, masked, barriers, forgotten, gpu_written, attachment, transfer;
static uint32_t layers[8], masks[8], values[8], samples_seen[8];
static VkImageAspectFlags aspect_seen;
static VkRect2D rect = {{2, 1}, {3, 4}};
static VkRect2D rect_seen;
static VkClearDepthStencilValue clear_seen;
#define CHECK(c) do { checks++; if (!(c)) { failures++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)
static void resolve_rt(VkExec *ex) { ex->rt_resolved = 1; }
static void warn_once(int key, const char *fmt, ...) { (void)key; (void)fmt; }
VkRect2D vk_exec_scissor(const VkExec *ex) { (void)ex; return rect; }
void vk_exec_barrier(VkExec *ex) { barriers++; ex->in_pass = 0; }
void vk_exec_pass_begin(VkExec *ex) { ex->in_pass = 1; }
void vk_exec_draw_new_cb(void) { forgotten++; }
void vk_texture_gpu_written(const tex_t *t) { (void)t; gpu_written++; }
static VkImageView view_of(VkExec *ex, const VkExecTarget *t) {
    (void)ex;
    CHECK(t->layers == 1 && t->view_type == VK_IMAGE_VIEW_TYPE_2D);
    return (VkImageView)(uintptr_t)(100 + t->base_layer);
}
void vk_clear_stencil_masked(VkCommandBuffer cb, VkFormat fmt, uint32_t samples, VkImageAspectFlags aspect,
                             VkImageView view, uint32_t w, uint32_t h, VkRect2D sc, uint32_t mask, uint32_t value) {
    CHECK(cb == (VkCommandBuffer)1 && fmt == VK_FORMAT_D32_SFLOAT_S8_UINT && w == 8 && h == 6);
    CHECK(aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
    layers[masked] = (uint32_t)(uintptr_t)view - 100;
    masks[masked] = mask, values[masked] = value, samples_seen[masked] = samples;
    masked++;
    rect_seen = sc;
}
void rc_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t na, const VkClearAttachment *a,
                              uint32_t nr, const VkClearRect *r) {
    (void)cb; CHECK(na == 1 && nr == 1); CHECK(r->layerCount == 3);
    attachment++; aspect_seen = a->aspectMask; clear_seen = a->clearValue.depthStencil;
}
void rc_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage image, VkImageLayout layout,
                                    const VkClearDepthStencilValue *v, uint32_t n,
                                    const VkImageSubresourceRange *r) {
    (void)cb; (void)image; CHECK(layout == VK_IMAGE_LAYOUT_GENERAL && n == 1);
    CHECK(r->baseArrayLayer == 2 && r->layerCount == 3);
    transfer++; aspect_seen = r->aspectMask; clear_seen = *v;
}

/* production functions are inserted here by stencil_clear_test.py */

static VkExec state(void) {
    masked = barriers = forgotten = gpu_written = attachment = transfer = 0;
    aspect_seen = 0;
    memset(&clear_seen, 0, sizeof clear_seen);
    rect = (VkRect2D){{2, 1}, {3, 4}};
    return (VkExec){.cb = (VkCommandBuffer)1, .pass_w = 8, .pass_h = 6, .pass_layers = 3, .has_scissor = 1,
        .scissor = {.x = 2, .y = 1, .w = 3, .h = 4},
        .depth = {.image = (VkImage)2, .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
                  .aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
                  .width = 8, .height = 6, .layers = 3, .base_layer = 2, .samples = 4}};
}

int main(void) {
    VkExec ex = state();
    p_clear_ds c = {.depth = 0.7f, .stencil = 0xa7, .stencil_mask = 0x15};
    clear_ds(&ex, &c);
    CHECK(masked == 3 && attachment == 0 && transfer == 0);
    CHECK(layers[0] == 2 && layers[1] == 3 && layers[2] == 4);
    CHECK(masks[0] == 0x15 && values[2] == 0xa7 && samples_seen[1] == 4);
    CHECK(!memcmp(&rect, &rect_seen, sizeof rect));
    CHECK(forgotten == 1 && ex.dirty == VK_EXEC_DIRTY_ALL && barriers == 2 && gpu_written == 1);

    ex = state(); c.depth_mask = 1;
    clear_ds(&ex, &c);
    CHECK(masked == 3 && attachment == 1 && aspect_seen == VK_IMAGE_ASPECT_DEPTH_BIT);
    CHECK(clear_seen.depth == 0.7f && forgotten == 1);

    ex = state(); c.stencil_mask = 0xff;
    clear_ds(&ex, &c);
    CHECK(masked == 0 && attachment == 1 && transfer == 0 && forgotten == 0);
    CHECK(aspect_seen == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));

    ex = state(); ex.has_scissor = 0;
    clear_ds(&ex, &c);
    CHECK(masked == 0 && transfer == 1 && attachment == 0);

    ex = state(); c.depth_mask = 0; c.stencil_mask = 0;
    clear_ds(&ex, &c);
    CHECK(!masked && !attachment && !transfer && !forgotten);

    ex = state(); ex.render_disabled = 1; c.stencil_mask = 1;
    clear_ds(&ex, &c);
    CHECK(!masked && !attachment && !transfer && !forgotten);

    ex = state(); rect.extent.width = 0;
    clear_ds(&ex, &c);
    CHECK(!masked && !attachment && !transfer && !forgotten);

    ex = state(); ex.depth.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    clear_ds(&ex, &c);
    CHECK(!masked && !attachment && !transfer && !forgotten);

    ex = state(); c.depth_mask = 1; c.stencil_mask = 0x100; c.depth = 2;
    clear_ds(&ex, &c);
    CHECK(!masked && attachment == 1 && aspect_seen == VK_IMAGE_ASPECT_DEPTH_BIT && clear_seen.depth == 1);
    printf("stencil-clear: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
