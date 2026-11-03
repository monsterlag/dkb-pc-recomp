// Original CPU-only checks of the real ResourceManager and backend gate.
using System;
using System.Collections.Generic;
using System.Reflection;
using Ryujinx.Graphics.Shader;
using Ryujinx.Graphics.Shader.Translation;

sealed class Accessor : IGpuAccessor
{
    int next = 1;
    public bool Capable;
    // Original minimal compute program: scheduler, unconditional EXIT, padding.
    readonly ulong[] code = { 0x001f8000fc0007e0, 0xe30000000007000f, 0, 0 };
    public ReadOnlySpan<ulong> GetCode(ulong address, int minimumSize) => code.AsSpan((int)(address / 8));
    public SetBindingPair CreateConstantBufferBinding(int index) => new(0, next++);
    public SetBindingPair CreateStorageBufferBinding(int index) => new(0, next++);
    public SetBindingPair CreateImageBinding(int count, bool isBuffer) => throw new Exception("unused image path");
    public SetBindingPair CreateTextureBinding(int count, bool isBuffer) => throw new Exception("unused texture path");
    public int QuerySamplerArrayLengthFromPool() => throw new Exception("unused sampler path");
    public int QueryTextureArrayLengthFromBuffer(int slot) => throw new Exception("unused texture path");
    public int QueryTextureArrayLengthFromPool() => throw new Exception("unused texture path");
    public void Log(string message) => throw new Exception("unexpected fallback: " + message);
    public bool QueryStorageBufferRuntimeCandidates() => true;
    public bool QueryHostSupportsStorageBufferArrayLength() => Capable;
}

static class Probe
{
    static int checks;
    static void Check(bool condition, string message)
    {
        checks++;
        if (!condition) throw new Exception(message);
    }

    static void CheckSignedAddressEmitter(Assembly assembly, object manager)
    {
        var accessor = new Accessor { Capable = true };
        var translator = Translator.CreateContext(0, accessor,
            new TranslationOptions(TargetLanguage.Spirv, TargetApi.Vulkan, TranslationFlags.Compute));
        Type emitterType = assembly.GetType("Ryujinx.Graphics.Shader.Translation.EmitterContext", true);
        Type registerType = assembly.GetType("Ryujinx.Graphics.Shader.Decoders.Register", true);
        Type registerKind = assembly.GetType("Ryujinx.Graphics.Shader.Decoders.RegisterType", true);
        object register = Activator.CreateInstance(registerType, 0, Enum.Parse(registerKind, "Gpr"));
        MethodInfo addressEmitter = assembly.GetType("Ryujinx.Graphics.Shader.Instructions.InstEmit", true)
            .GetMethod("Get40BitsAddress", BindingFlags.Static | BindingFlags.NonPublic);
        foreach (var test in new[] { (0x200000100UL, 16), (0x2fffffff0UL, 32),
                                     (0x200000100UL, -16), (0x300000010UL, -32) })
        {
            object emitter = Activator.CreateInstance(emitterType, translator, manager, null, false, false);
            object pair = addressEmitter.Invoke(null, new object[] { emitter, register, true, test.Item2 });
            var values = new Dictionary<object, uint>();
            uint Read(object operand)
            {
                if (values.TryGetValue(operand, out uint value)) return value;
                Type type = operand.GetType();
                string kind = type.GetProperty("Type").GetValue(operand).ToString();
                int data = (int)type.GetProperty("Value").GetValue(operand);
                if (kind == "Constant") return unchecked((uint)data);
                if (kind == "Register") return (data & 0xffffff) == 0 ? (uint)test.Item1 : (uint)(test.Item1 >> 32);
                throw new Exception("unexpected address-emitter operand " + kind);
            }
            // Interpret the actual emitted three-op integer subset. This is not
            // a reimplementation of the address formula under test.
            foreach (object operation in (Array)emitterType.GetMethod("GetOperations").Invoke(emitter, null))
            {
                Type type = operation.GetType();
                string inst = type.GetProperty("Inst").GetValue(operation).ToString();
                object Source(int i) => type.GetMethod("GetSource").Invoke(operation, new object[] { i });
                uint value = inst switch
                {
                    "Add" => unchecked(Read(Source(0)) + Read(Source(1))),
                    "CompareLessU32" => Read(Source(0)) < Read(Source(1)) ? uint.MaxValue : 0,
                    "ConditionalSelect" => Read(Source(0)) != 0 ? Read(Source(1)) : Read(Source(2)),
                    _ => throw new Exception("unexpected address-emitter instruction " + inst),
                };
                values[type.GetProperty("Dest").GetValue(operation)] = value;
            }
            uint low = Read(pair.GetType().GetField("Item1").GetValue(pair));
            uint high = Read(pair.GetType().GetField("Item2").GetValue(pair));
            ulong actual = low | (ulong)high << 32;
            ulong expected = unchecked((ulong)((long)test.Item1 + test.Item2));
            Check(actual == expected, $"signed effective address {test.Item1:x}+({test.Item2})");
        }
    }

    static int Main()
    {
        Assembly assembly = typeof(IGpuAccessor).Assembly;
        Type managerType = assembly.GetType("Ryujinx.Graphics.Shader.Translation.ResourceManager", true);
        object manager = Activator.CreateInstance(managerType, new object[] { ShaderStage.Compute, new Accessor(), null });
        MethodInfo bind = managerType.GetMethod("TryGetStorageBufferBinding");
        int Bind(bool guarded, bool write)
        {
            object[] args = { 0, 4, write, 0, guarded };
            Check((bool)bind.Invoke(manager, args), "binding allocation failed");
            return (int)args[3];
        }
        int guardedBinding = Bind(true, false);
        int legacyBinding = Bind(false, false);
        Check(guardedBinding != legacyBinding, "guarded and legacy descriptors alias");
        Check(Bind(true, true) == guardedBinding, "guarded write changed descriptor identity");
        var descriptors = (BufferDescriptor[])managerType.GetMethod("GetStorageBufferDescriptors").Invoke(manager, null);
        Check(descriptors.Length == 2, "same cbuf pair requires two independent contracts");
        foreach (var descriptor in descriptors)
        {
            Check(descriptor.SbCbSlot == 0 && descriptor.SbCbOffset == 4, "reflection pair changed");
            Check((int)descriptor.Flags == (descriptor.Binding == guardedBinding ? 0x81 : 0), "guarded/write flags leaked");
        }

        Type pass = assembly.GetType("Ryujinx.Graphics.Shader.Translation.Optimizations.GlobalToStorage", true);
        Type block = assembly.GetType("Ryujinx.Graphics.Shader.IntermediateRepresentation.BasicBlock", true);
        foreach (var test in new[] { (TargetLanguage.Glsl, true), (TargetLanguage.Spirv, false) })
        {
            try
            {
                pass.GetMethod("RunPass").Invoke(null, new object[] { null, Array.CreateInstance(block, 0), null,
                    new Accessor { Capable = test.Item2 }, test.Item1 });
                throw new Exception("unsupported backend accepted guarded candidates");
            }
            catch (TargetInvocationException e) when (e.InnerException is NotSupportedException)
            {
                Check(true, "backend gate rejected before any pass mutation");
            }
        }
        CheckSignedAddressEmitter(assembly, manager);
        Console.WriteLine($"resource/capability probe: {checks} checks");
        return 0;
    }
}
