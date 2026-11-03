// Headless compute readback for original storage_candidates_gpu_test.py fixtures.
// Uniform address tokens are integers, never Vulkan device addresses. No game or window inputs.
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(2); } while (0)
#define VK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) FAIL("%s: VkResult %d", #call, (int)r_); } while (0)
enum { BUFFERS = 5, MAX_BINDINGS = 32, PACKET_MAGIC = 0x53434750 };
typedef struct { VkBuffer buffer; VkDeviceMemory memory; void *mapped; uint32_t size; } Buffer;
typedef struct { uint32_t binding, type, buffer, offset, range; } Binding;
static atomic_uint errors, warnings;

static VKAPI_ATTR VkBool32 VKAPI_CALL validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)type; (void)user;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) atomic_fetch_add(&errors, 1);
    else atomic_fetch_add(&warnings, 1);
    fprintf(stderr, "Vulkan %s: %s: %s\n", severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning",
            data->pMessageIdName ? data->pMessageIdName : "unnamed", data->pMessage);
    return VK_FALSE;
}

static void json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\'), putchar(*p);
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}

static uint8_t *read_file(const char *name, size_t *size) {
    FILE *f = fopen(name, "rb");
    if (!f || fseek(f, 0, SEEK_END)) FAIL("cannot open %s", name);
    long n = ftell(f);
    if (n <= 0 || n > 1048576 || fseek(f, 0, SEEK_SET)) FAIL("unsupported input length: %s", name);
    uint8_t *data = malloc((size_t)n);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n || fclose(f)) FAIL("cannot read %s", name);
    *size = (size_t)n;
    return data;
}

static uint32_t word(const uint8_t **p, const uint8_t *end) {
    uint32_t n;
    if ((size_t)(end - *p) < 4) FAIL("truncated owned fixture packet");
    memcpy(&n, *p, 4); *p += 4;
    return n;
}

static Buffer make_buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *memory, uint32_t size) {
    Buffer b = {.size = size};
    VkBufferCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VK(vkCreateBuffer(device, &ci, NULL, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    uint32_t type = UINT32_MAX;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memory->memoryTypeCount; i++)
        if ((req.memoryTypeBits & (1u << i)) && (memory->memoryTypes[i].propertyFlags & want) == want) { type = i; break; }
    if (type == UINT32_MAX) FAIL("fixture requires coherent host-visible memory");
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = type};
    VK(vkAllocateMemory(device, &ai, NULL, &b.memory));
    VK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    VK(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped));
    return b;
}

int main(int argc, char **argv) {
    if (argc != 6 || (strcmp(argv[5], "0") && strcmp(argv[5], "1")))
        FAIL("usage: runner shader.spv packet.bin readback.bin device-name validation0or1");
    int validate = !strcmp(argv[5], "1");
    size_t shader_size, packet_size;
    uint8_t *shader = read_file(argv[1], &shader_size), *packet = read_file(argv[2], &packet_size);
    if (shader_size < 20 || shader_size % 4 || ((uint32_t *)shader)[0] != 0x07230203) FAIL("invalid owned SPIR-V input");
    const uint8_t *p = packet, *end = packet + packet_size;
    if (word(&p, end) != PACKET_MAGIC || word(&p, end) != 1) FAIL("unsupported fixture packet");
    uint32_t count = word(&p, end);
    if (!count || count > MAX_BINDINGS || word(&p, end) != BUFFERS) FAIL("unsupported descriptor count");
    uint32_t features = word(&p, end), subgroup_mask = word(&p, end), alignment = word(&p, end);
    if ((features & ~7u) || (subgroup_mask & ~7u) || (alignment != 256 && alignment != 0) || word(&p, end))
        FAIL("unsupported fixture requirements");
    uint32_t sizes[BUFFERS];
    for (unsigned i = 0; i < BUFFERS; i++) {
        sizes[i] = word(&p, end);
        if (sizes[i] < 512 || sizes[i] > 131072) FAIL("unsupported fixture buffer size");
    }
    Binding bs[MAX_BINDINGS];
    uint32_t ubos = 0, ssbos = 0;
    for (uint32_t i = 0; i < count; i++) {
        bs[i].binding = word(&p, end);
        bs[i].type = word(&p, end);
        bs[i].buffer = word(&p, end);
        bs[i].offset = word(&p, end);
        bs[i].range = word(&p, end);
        Binding *b = &bs[i];
        if (b->binding >= 128 || b->type > 1 || b->buffer >= BUFFERS || !b->range ||
            b->offset > sizes[b->buffer] || b->range > sizes[b->buffer] - b->offset ||
            (!b->type && b->buffer < 3 && !(b->binding == 0 && b->buffer == 2)) ||
            (b->type && b->buffer >= 3)) FAIL("unsupported owned binding");
        for (uint32_t j = 0; j < i; j++) if (bs[j].binding == b->binding) FAIL("duplicate descriptor binding");
        if (b->type) ssbos++; else ubos++;
    }
    const uint8_t *initial[BUFFERS];
    for (unsigned i = 0; i < BUFFERS; i++) {
        if ((size_t)(end - p) < sizes[i]) FAIL("truncated fixture buffer data");
        initial[i] = p; p += sizes[i];
    }
    if (p != end) FAIL("unexpected fixture packet suffix");

    const char *layer = "VK_LAYER_KHRONOS_validation", *debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    uint32_t layer_count = 0, extension_count = 0;
    VK(vkEnumerateInstanceLayerProperties(&layer_count, NULL));
    if (layer_count > 256) FAIL("unexpected instance layer count");
    VkLayerProperties layers[256];
    VK(vkEnumerateInstanceLayerProperties(&layer_count, layers));
    int layer_available = 0, debug_available = 0;
    for (uint32_t i = 0; i < layer_count; i++) if (!strcmp(layers[i].layerName, layer)) layer_available = 1;
    VK(vkEnumerateInstanceExtensionProperties(NULL, &extension_count, NULL));
    if (extension_count > 256) FAIL("unexpected instance extension count");
    VkExtensionProperties extensions[256];
    VK(vkEnumerateInstanceExtensionProperties(NULL, &extension_count, extensions));
    for (uint32_t i = 0; i < extension_count; i++) if (!strcmp(extensions[i].extensionName, debug_extension)) debug_available = 1;
    if (validate && (!layer_available || !debug_available)) FAIL("requested Khronos validation/debug messenger unavailable");
    VkDebugUtilsMessengerCreateInfoEXT debug = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, .pfnUserCallback = validation};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "Original storage candidate regression",
                            .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
        .pNext = validate ? &debug : NULL, .enabledLayerCount = validate ? 1u : 0u, .ppEnabledLayerNames = &layer,
        .enabledExtensionCount = validate ? 1u : 0u, .ppEnabledExtensionNames = &debug_extension};
    VkInstance instance;
    VK(vkCreateInstance(&ici, NULL, &instance));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT make = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (!make) FAIL("requested debug messenger function unavailable");
        VK(make(instance, &debug, NULL, &messenger));
    }
    uint32_t physical_count = 0;
    VK(vkEnumeratePhysicalDevices(instance, &physical_count, NULL));
    if (!physical_count || physical_count > 32) FAIL("unsupported physical-device count");
    VkPhysicalDevice physicals[32], physical = VK_NULL_HANDLE;
    VK(vkEnumeratePhysicalDevices(instance, &physical_count, physicals));
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < physical_count && !physical; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicals[i], &properties);
        if (*argv[4] && !strstr(properties.deviceName, argv[4])) continue;
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[i], &n, NULL);
        if (n > 64) FAIL("unsupported queue-family count");
        VkQueueFamilyProperties qs[64];
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[i], &n, qs);
        for (uint32_t j = 0; j < n; j++) if (qs[j].queueCount && (qs[j].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            physical = physicals[i]; family = j; break;
        }
    }
    if (!physical) FAIL("no compute device matching '%s'", argv[4]);
    VkPhysicalDeviceSubgroupProperties subgroup = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, .pNext = &subgroup};
    VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
    vkGetPhysicalDeviceProperties2(physical, &properties);
    if (properties.properties.apiVersion < VK_API_VERSION_1_3) FAIL("fixture requires Vulkan1.3");
    const VkPhysicalDeviceLimits *limits = &properties.properties.limits;
    for (uint32_t i = 0; i < count; i++) {
        VkDeviceSize align = bs[i].type ? limits->minStorageBufferOffsetAlignment : limits->minUniformBufferOffsetAlignment;
        uint32_t max = bs[i].type ? limits->maxStorageBufferRange : limits->maxUniformBufferRange;
        if ((align && bs[i].offset % align) || bs[i].range > max) FAIL("device cannot represent owned descriptor range/alignment");
    }
    VkSubgroupFeatureFlags wanted = (subgroup_mask & 1 ? VK_SUBGROUP_FEATURE_VOTE_BIT : 0) |
        (subgroup_mask & 2 ? VK_SUBGROUP_FEATURE_BALLOT_BIT : 0) | (subgroup_mask & 4 ? VK_SUBGROUP_FEATURE_SHUFFLE_BIT : 0);
    if (wanted && (!(subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) || (subgroup.supportedOperations & wanted) != wanted))
        FAIL("declared compute subgroup capabilities unsupported");
    VkPhysicalDeviceFeatures available;
    vkGetPhysicalDeviceFeatures(physical, &available);
    if (!available.robustBufferAccess) FAIL("robustBufferAccess is required for every baseline and candidate run");
    if ((features & 1 && !available.shaderFloat64) || (features & 2 && !available.shaderInt64) ||
        (features & 4 && !available.shaderImageGatherExtended)) FAIL("declared shader feature unsupported");
    VkPhysicalDeviceFeatures enabled = {.robustBufferAccess = VK_TRUE, .shaderFloat64 = !!(features & 1),
        .shaderInt64 = !!(features & 2), .shaderImageGatherExtended = !!(features & 4)};
    float priority = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
        .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qi, .pEnabledFeatures = &enabled};
    VkDevice device;
    VK(vkCreateDevice(physical, &di, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    Buffer buffers[BUFFERS];
    for (unsigned i = 0; i < BUFFERS; i++) {
        buffers[i] = make_buffer(device, &memory, sizes[i]);
        memcpy(buffers[i].mapped, initial[i], sizes[i]);
    }
    VkDescriptorSetLayoutBinding bindings[MAX_BINDINGS];
    for (uint32_t i = 0; i < count; i++) bindings[i] = (VkDescriptorSetLayoutBinding){.binding = bs[i].binding,
        .descriptorType = bs[i].type ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = count, .pBindings = bindings};
    VkDescriptorSetLayout layout;
    VK(vkCreateDescriptorSetLayout(device, &li, NULL, &layout));
    VkDescriptorPoolSize ps[2]; uint32_t pn = 0;
    if (ubos) ps[pn++] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, ubos};
    if (ssbos) ps[pn++] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ssbos};
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
        .poolSizeCount = pn, .pPoolSizes = ps};
    VkDescriptorPool pool;
    VK(vkCreateDescriptorPool(device, &dpi, NULL, &pool));
    VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout};
    VkDescriptorSet set;
    VK(vkAllocateDescriptorSets(device, &dai, &set));
    VkDescriptorBufferInfo ranges[MAX_BINDINGS]; VkWriteDescriptorSet writes[MAX_BINDINGS];
    for (uint32_t i = 0; i < count; i++) {
        ranges[i] = (VkDescriptorBufferInfo){buffers[bs[i].buffer].buffer, bs[i].offset, bs[i].range};
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
            .dstBinding = bs[i].binding, .descriptorCount = 1, .descriptorType = bindings[i].descriptorType,
            .pBufferInfo = &ranges[i]};
    }
    vkUpdateDescriptorSets(device, count, writes, 0, NULL);
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &layout};
    VkPipelineLayout pipeline_layout;
    VK(vkCreatePipelineLayout(device, &pli, NULL, &pipeline_layout));
    VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shader_size, .pCode = (const uint32_t *)shader};
    VkShaderModule module;
    VK(vkCreateShaderModule(device, &smi, NULL, &module));
    VkComputePipelineCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = pipeline_layout, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}};
    VkPipeline pipeline;
    VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline));
    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
    VkCommandPool command_pool;
    VK(vkCreateCommandPool(device, &cpi, NULL, &command_pool));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer command;
    VK(vkAllocateCommandBuffers(device, &cai, &command));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK(vkBeginCommandBuffer(command, &bi));
    VkMemoryBarrier before = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, NULL, 0, NULL);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(command, 1, 1, 1);
    VkMemoryBarrier after = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, NULL, 0, NULL);
    VK(vkEndCommandBuffer(command));
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence;
    VK(vkCreateFence(device, &fi, NULL, &fence));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
    VK(vkQueueSubmit(queue, 1, &si, fence));
    VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(30000000000)));
    FILE *out = fopen(argv[3], "wb");
    if (!out) FAIL("cannot open readback output");
    for (unsigned i = 0; i < BUFFERS; i++)
        if (fwrite(buffers[i].mapped, 1, sizes[i], out) != sizes[i]) FAIL("cannot write readback buffer");
    if (fclose(out)) FAIL("cannot close readback output");
    VK(vkDeviceWaitIdle(device));
    vkDestroyFence(device, fence, NULL); vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL); vkDestroyShaderModule(device, module, NULL);
    vkDestroyPipelineLayout(device, pipeline_layout, NULL); vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyDescriptorSetLayout(device, layout, NULL);
    for (unsigned i = 0; i < BUFFERS; i++) {
        vkUnmapMemory(device, buffers[i].memory); vkDestroyBuffer(device, buffers[i].buffer, NULL);
        vkFreeMemory(device, buffers[i].memory, NULL);
    }
    vkDestroyDevice(device, NULL);
    if (messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(instance, messenger, NULL);
    }
    vkDestroyInstance(instance, NULL);
    printf("{\"device\":"); json_string(properties.properties.deviceName);
    printf(",\"driver\":"); json_string(driver.driverName);
    printf(",\"driver_info\":"); json_string(driver.driverInfo);
    printf(",\"api_version\":%u,\"driver_version\":%u,\"vendor_id\":%u,\"device_id\":%u,\"queue_family\":%u",
        properties.properties.apiVersion, properties.properties.driverVersion, properties.properties.vendorID,
        properties.properties.deviceID, family);
    printf(",\"robust_buffer_access_supported\":true,\"robust_buffer_access_enabled\":true,\"required_feature_mask\":%u",
        features);
    printf(",\"host_address_alignment\":%u", alignment);
    printf(",\"validation_available\":%s,\"validation_requested\":%s,\"validation_explicit_layer_requested\":%s,\"debug_messenger_created\":%s",
        layer_available ? "true" : "false", validate ? "true" : "false", validate ? "true" : "false", messenger ? "true" : "false");
    printf(",\"validation_errors\":%u,\"validation_warnings\":%u,\"bindings\":%u}\n",
        atomic_load(&errors), atomic_load(&warnings), count);
    free(shader); free(packet);
    return atomic_load(&errors) ? 1 : 0;
}
