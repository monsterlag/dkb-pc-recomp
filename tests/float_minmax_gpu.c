// Headless runner for the owned float_minmax_gpu_test.py compute adapter. No surface or game inputs.
// The Python adapter supplies reflected bindings/range; this runner reports device float controls without
// enabling a different shader environment. Numerical target checks happen in Python on the raw readback bits.
#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

#define FAIL(...) do { fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); exit(2); } while (0)
#define VK(call) do { VkResult result_ = (call); if (result_ != VK_SUCCESS) \
    FAIL("%s failed: VkResult %d", #call, (int)result_); } while (0)

static atomic_uint validation_errors;
static VKAPI_ATTR VkBool32 VKAPI_CALL validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)type; (void)user;
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) atomic_fetch_add(&validation_errors, 1);
    fprintf(stderr, "Vulkan %s: %s: %s\n", severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? "error" : "warning",
            data->pMessageIdName ? data->pMessageIdName : "unnamed", data->pMessage);
    return VK_FALSE;
}

static void json_string(const char *text) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"' || *p == '\\') putchar('\\'), putchar(*p);
        else if (*p < 32) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}

static uint32_t number(const char *text) {
    char *end;
    unsigned long value = strtoul(text, &end, 10);
    if (!*text || *end || value > UINT32_MAX) FAIL("invalid integer argument: %s", text);
    return (uint32_t)value;
}

static void *read_bytes(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) FAIL("cannot open input %s", path);
    long length = ftell(file);
    if (length <= 0 || length > 65536 || fseek(file, 0, SEEK_SET)) FAIL("unexpected input size: %s", path);
    void *bytes = malloc((size_t)length);
    if (!bytes || fread(bytes, 1, (size_t)length, file) != (size_t)length || fclose(file)) FAIL("cannot read %s", path);
    *size = (size_t)length;
    return bytes;
}

typedef struct { VkBuffer buffer; VkDeviceMemory memory; void *mapped; VkDeviceSize size; } Buffer;
static Buffer buffer(VkDevice device, const VkPhysicalDeviceMemoryProperties *properties,
                     VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer result = {.size = size};
    VkBufferCreateInfo create = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VK(vkCreateBuffer(device, &create, NULL, &result.buffer));
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < properties->memoryTypeCount; i++)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (properties->memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
    if (type == UINT32_MAX) FAIL("no coherent host-visible memory type for probe buffer");
    VkMemoryAllocateInfo allocation = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                      .allocationSize = requirements.size, .memoryTypeIndex = type};
    VK(vkAllocateMemory(device, &allocation, NULL, &result.memory));
    VK(vkBindBufferMemory(device, result.buffer, result.memory, 0));
    VK(vkMapMemory(device, result.memory, 0, size, 0, &result.mapped));
    memset(result.mapped, 0, (size_t)size);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 10) FAIL("usage: runner shader.spv inputs.bin outputs.bin input_binding output_binding UBO_bytes FP_modes device_name validation0or1");
    uint32_t input_binding = number(argv[4]), output_binding = number(argv[5]), uniform_bytes = number(argv[6]);
    uint32_t requirements = number(argv[7]), modes = requirements & 31, int64 = !!(requirements & 256), validate = number(argv[9]);
    if (input_binding == output_binding || input_binding >= 32 || output_binding >= 32 ||
        uniform_bytes < 16 || uniform_bytes > 65536 || (requirements & ~287u) || validate > 1) FAIL("unsupported fixture metadata");
    size_t shader_size, input_size;
    uint32_t *shader = read_bytes(argv[1], &shader_size), *inputs = read_bytes(argv[2], &input_size);
    if (shader_size % 4 || shader_size < 20 || shader[0] != 0x07230203 || input_size != 9 * 16) FAIL("unexpected fixture inputs");
    const char *layer = "VK_LAYER_KHRONOS_validation", *extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    VkDebugUtilsMessengerCreateInfoEXT debug = {.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, .pfnUserCallback = validation};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "Owned float min/max regression",
                            .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo create = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
        .pNext = validate ? &debug : NULL, .enabledLayerCount = validate, .ppEnabledLayerNames = &layer,
        .enabledExtensionCount = validate, .ppEnabledExtensionNames = &extension};
    VkInstance instance;
    VK(vkCreateInstance(&create, NULL, &instance));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT make = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (!make) FAIL("requested validation messenger unavailable");
        VK(make(instance, &debug, NULL, &messenger));
    }
    uint32_t count = 0;
    VK(vkEnumeratePhysicalDevices(instance, &count, NULL));
    if (!count || count > 32) FAIL("unexpected physical-device count %u", count);
    VkPhysicalDevice physicals[32], physical = VK_NULL_HANDLE;
    VK(vkEnumeratePhysicalDevices(instance, &count, physicals));
    uint32_t queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < count && !physical; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicals[i], &properties);
        if (argv[8][0] && !strstr(properties.deviceName, argv[8])) continue;
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[i], &families, NULL);
        if (families > 64) FAIL("unexpected queue-family count");
        VkQueueFamilyProperties queues[64];
        vkGetPhysicalDeviceQueueFamilyProperties(physicals[i], &families, queues);
        for (uint32_t j = 0; j < families; j++) if (queues[j].queueCount && (queues[j].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            physical = physicals[i]; queue_family = j; break;
        }
    }
    if (!physical) FAIL("no compute device matching requested name '%s'", argv[8]);
    VkPhysicalDeviceFloatControlsProperties floats = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, .pNext = &floats};
    VkPhysicalDeviceProperties2 properties = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driver};
    vkGetPhysicalDeviceProperties2(physical, &properties);
    if (properties.properties.apiVersion < VK_API_VERSION_1_2 || properties.properties.limits.maxUniformBufferRange < uniform_bytes)
        FAIL("device cannot represent fixture's Vulkan1.2/UBO requirements");
    VkBool32 supported[5] = {floats.shaderDenormPreserveFloat32, floats.shaderDenormFlushToZeroFloat32,
        floats.shaderSignedZeroInfNanPreserveFloat32, floats.shaderRoundingModeRTEFloat32, floats.shaderRoundingModeRTZFloat32};
    for (unsigned i = 0; i < 5; i++) if ((modes & (1u << i)) && !supported[i]) FAIL("emitted FP32 mode%u unsupported", i);
    VkPhysicalDeviceFeatures available, enabled = {.shaderInt64 = int64};
    vkGetPhysicalDeviceFeatures(physical, &available);
    if (int64 && !available.shaderInt64) FAIL("emitted integer64 arithmetic unsupported");
    uint32_t float_controls2 = 0;
#ifdef VK_KHR_shader_float_controls2
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR features2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features2};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    float_controls2 = features2.shaderFloatControls2;
#endif
    float priority = 1;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = queue_family,
                                  .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qi, .pEnabledFeatures = &enabled};
    VkDevice device;
    VK(vkCreateDevice(physical, &di, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, queue_family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    Buffer input = buffer(device, &memory, uniform_bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    Buffer output = buffer(device, &memory, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDescriptorSetLayoutBinding bindings[2] = {
        {.binding = input_binding, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = output_binding, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bindings};
    VkDescriptorSetLayout layout;
    VK(vkCreateDescriptorSetLayout(device, &li, NULL, &layout));
    VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = sizes};
    VkDescriptorPool pool;
    VK(vkCreateDescriptorPool(device, &pi, NULL, &pool));
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
                                     .descriptorSetCount = 1, .pSetLayouts = &layout};
    VkDescriptorSet set;
    VK(vkAllocateDescriptorSets(device, &ai, &set));
    VkDescriptorBufferInfo ranges[2] = {{input.buffer, 0, input.size}, {output.buffer, 0, output.size}};
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = input_binding, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &ranges[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = output_binding, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &ranges[1]}};
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &layout};
    VkPipelineLayout pipeline_layout;
    VK(vkCreatePipelineLayout(device, &pli, NULL, &pipeline_layout));
    VkShaderModuleCreateInfo sm = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = shader_size, .pCode = shader};
    VkShaderModule module;
    VK(vkCreateShaderModule(device, &sm, NULL, &module));
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pipeline_layout,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = module, .pName = "main"}};
    VkPipeline pipeline;
    VK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, NULL, &pipeline));
    VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = queue_family};
    VkCommandPool command_pool;
    VK(vkCreateCommandPool(device, &cpi, NULL, &command_pool));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer command;
    VK(vkAllocateCommandBuffers(device, &cai, &command));
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK(vkBeginCommandBuffer(command, &begin));
    VkMemoryBarrier before = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, NULL, 0, NULL);
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &set, 0, NULL);
    vkCmdDispatch(command, 1, 1, 1);
    VkMemoryBarrier after = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                             .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &after, 0, NULL, 0, NULL);
    VK(vkEndCommandBuffer(command));
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK(vkCreateFence(device, &fi, NULL, &fence));
    uint32_t results[36];
    for (unsigned row = 0; row < 9; row++) {
        memcpy(input.mapped, inputs + row * 4, 16);
        memset(output.mapped, 0xa5, 16);
        VK(vkResetFences(device, 1, &fence));
        VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &command};
        VK(vkQueueSubmit(queue, 1, &submit, fence));
        VK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(30000000000)));
        memcpy(results + row * 4, output.mapped, 16);
    }
    VK(vkDeviceWaitIdle(device));
    FILE *file = fopen(argv[3], "wb");
    if (!file || fwrite(results, 1, sizeof results, file) != sizeof results || fclose(file)) FAIL("cannot write readback");
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, command_pool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyShaderModule(device, module, NULL);
    vkDestroyPipelineLayout(device, pipeline_layout, NULL);
    vkDestroyDescriptorPool(device, pool, NULL);
    vkDestroyDescriptorSetLayout(device, layout, NULL);
    vkUnmapMemory(device, input.memory); vkUnmapMemory(device, output.memory);
    vkDestroyBuffer(device, input.buffer, NULL); vkDestroyBuffer(device, output.buffer, NULL);
    vkFreeMemory(device, input.memory, NULL); vkFreeMemory(device, output.memory, NULL);
    vkDestroyDevice(device, NULL);
    if (messenger) {
        PFN_vkDestroyDebugUtilsMessengerEXT destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) destroy(instance, messenger, NULL);
    }
    vkDestroyInstance(instance, NULL);
    printf("{\"device\":"); json_string(properties.properties.deviceName);
    printf(",\"driver\":"); json_string(driver.driverName);
    printf(",\"driver_info\":"); json_string(driver.driverInfo);
    printf(",\"api_version\":%u,\"vendor_id\":%u,\"device_id\":%u,\"driver_version\":%u,\"queue_family\":%u",
           properties.properties.apiVersion, properties.properties.vendorID, properties.properties.deviceID,
           properties.properties.driverVersion, queue_family);
    printf(",\"fp32_properties\":{\"denorm_preserve\":%u,\"denorm_flush\":%u,\"signed_zero_inf_nan_preserve\":%u,\"round_rte\":%u,\"round_rtz\":%u}",
           supported[0], supported[1], supported[2], supported[3], supported[4]);
    printf(",\"float_controls2_supported\":%u,\"float_controls2_enabled\":false,\"emitted_fp_mode_mask\":%u,\"shader_int64_enabled\":%u,\"validation_requested\":%u,\"validation_errors\":%u,\"readback_values\":36}\n",
           float_controls2, modes, int64, validate, atomic_load(&validation_errors));
    free(shader); free(inputs);
    return atomic_load(&validation_errors) ? 1 : 0;
}
