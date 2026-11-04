"""Original Maxwell encoders and small CPU evaluators for storage-candidate tests.

No captured shader bytes or guest memory are used. Tests and the separate GPU
helper import these reusable scalar/vector fixtures; this module runs no tests.
"""
import struct

PT = 7 << 16
SCHED = 0x001F8000FC0007E0
EXIT = 0xE30000000000000F | PT
MASK32 = (1 << 32) - 1
WIDE_VALUES = (0x13579BDF, 0x2468ACE0, 0xDEADBEEF, 0x10203040)


# Encode the named Maxwell fields consumed by InstDecoders and InstEmitMemory.
# These instructions were authored for this test; no captured shader is used.
def movc(dst, bank, word):
    return 0x4C98000000000000 | (15 << 39) | (bank << 34) | (word << 20) | PT | dst


def movi(dst, value):
    return 0x0100000000000000 | (value << 20) | (15 << 12) | PT | dst


def iaddc(dst, a, bank, word, cc=False, carry=False):
    return (0x4C10000000000000 | (bank << 34) | (word << 20) | (a << 8) | PT | dst |
            (int(cc) << 47) | (int(carry) << 43))


def iaddr(dst, a, b, cc=False, carry=False):
    return (0x5C10000000000000 | (b << 20) | (a << 8) | PT | dst |
            (int(cc) << 47) | (int(carry) << 43))


def global_memory(store, value, address):
    # B32 (size4), E=1: the address occupies address/address+1.
    opcode = 0xEED8000000000000 if store else 0xEED0000000000000
    return opcode | (4 << 48) | (1 << 45) | (address << 8) | PT | value


def instructions(inline):
    ops = [movc(0, 0, 4), movc(1, 0, 5)]
    if inline:
        ops += [iaddc(0, 0, 1, 0, cc=True), iaddc(1, 1, 1, 1, carry=True)]
    else:
        ops += [movi(6, 32), movi(7, 0), iaddr(0, 0, 6, cc=True), iaddr(1, 1, 7, carry=True)]
    return ops + [global_memory(False, 2, 0), movc(4, 0, 8), movc(5, 0, 9),
                  global_memory(True, 2, 4), EXIT]


def shader_bytes(inline):
    ops = instructions(inline)
    words = []
    for i in range(0, len(ops), 3):
        batch = ops[i:i + 3]
        words += [SCHED] + batch + [0] * (3 - len(batch))
    return struct.pack('<' + 'Q' * len(words), *words)


def wide_instructions(bits, inline=True, memory_offset=0, scalar_offset=32):
    """Original aligned vector copy, with data only in R8..R11.

    Input address R0/R1 adds the scalar; output R4/R5 is independent. The same
    immediate memory_offset applies to both vector operations. R6/R7 remain
    literal temporaries, so a four-word load never overwrites either pointer.
    """
    if bits not in (64, 128) or not -(1 << 23) <= memory_offset < (1 << 23):
        raise ValueError('expected B64/B128 and a signed 24-bit memory offset')
    size = 5 if bits == 64 else 6
    ops = [movc(0, 0, 4), movc(1, 0, 5)]
    if inline:
        ops += [iaddc(0, 0, 1, 0, cc=True), iaddc(1, 1, 1, 1, carry=True)]
    else:
        ops += [movi(6, scalar_offset), movi(7, 0), iaddr(0, 0, 6, cc=True), iaddr(1, 1, 7, carry=True)]
    encoded_offset = (memory_offset & 0xffffff) << 20
    load = (global_memory(False, 8, 0) & ~(7 << 48)) | (size << 48) | encoded_offset
    store = (global_memory(True, 8, 4) & ~(7 << 48)) | (size << 48) | encoded_offset
    return ops + [load, movc(4, 0, 8), movc(5, 0, 9), store, EXIT]


def wide_shader_bytes(bits, inline=True, memory_offset=0, scalar_offset=32):
    ops = wide_instructions(bits, inline, memory_offset, scalar_offset)
    words = []
    for i in range(0, len(ops), 3):
        batch = ops[i:i + 3]
        words += [SCHED] + batch + [0] * (3 - len(batch))
    return struct.pack('<' + 'Q' * len(words), *words)


def c1_pointer_bytes(dynamic_offset):
    ops = [movc(0, 1, 0), movc(1, 1, 1)]
    if dynamic_offset:
        ops += [iaddc(0, 0, 0, 4, cc=True), iaddc(1, 1, 0, 5, carry=True)]
    ops += [global_memory(False, 2, 0), movc(4, 0, 8), movc(5, 0, 9),
            global_memory(True, 2, 4), EXIT]
    words = []
    for i in range(0, len(ops), 3):
        batch = ops[i:i + 3]
        words += [SCHED] + batch + [0] * (3 - len(batch))
    return struct.pack('<' + 'Q' * len(words), *words)


def loads_uniform_word_zero(data, binding):
    """Check actual emitted SPIR-V retains a load of the unknown c1 low word.

    The translator represents cbufs as a struct containing a vec4 array. Check
    the binding and constant access-chain indices rather than generated IDs.
    """
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    constants, bindings, chains, loaded = {}, {}, {}, set()
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 65535
        assert count and offset + count <= len(words)
        operands = words[offset + 1:offset + count]
        if opcode == 43 and len(operands) == 3:  # OpConstant, 32-bit literal
            constants[operands[1]] = operands[2]
        elif opcode == 71 and operands[1] == 33:  # OpDecorate Binding
            bindings[operands[0]] = operands[2]
        elif opcode == 65:  # OpAccessChain
            chains[operands[1]] = (operands[2], operands[3:])
        elif opcode == 61:  # OpLoad
            loaded.add(operands[2])
        offset += count
    return any(pointer in loaded and bindings.get(base) == binding and
               tuple(constants.get(index) for index in indices) == (0, 0, 0)
               for pointer, (base, indices) in chains.items())


def evaluate(inline, input_pointer, output_pointer):
    """Evaluate the original fixture's small integer subset, including carry.

    This checks the authored byte fields and intended address equivalence. The
    regression itself below uses the actual translator and emitted reflection.
    Addresses are dictionary keys; this function dereferences no host pointer.
    """
    cbuf = {(0, 4): input_pointer & MASK32, (0, 5): input_pointer >> 32,
            (0, 8): output_pointer & MASK32, (0, 9): output_pointer >> 32,
            (1, 0): 32, (1, 1): 0}
    memory = {input_pointer + 32: 0x1234ABCD}
    regs, carry = {}, 0
    for op in instructions(inline):
        dst, a = op & 255, (op >> 8) & 255
        prefix = op >> 51
        if op == EXIT:
            return memory[output_pointer]
        if prefix == 0x4C98000000000000 >> 51:
            regs[dst] = cbuf[((op >> 34) & 31, (op >> 20) & 0x3FFF)]
        elif op >> 56 == 1:
            regs[dst] = (op >> 20) & MASK32
        elif prefix in (0x4C10000000000000 >> 51, 0x5C10000000000000 >> 51):
            if prefix == 0x4C10000000000000 >> 51:
                b = cbuf[((op >> 34) & 31, (op >> 20) & 0x3FFF)]
            else:
                b = regs[(op >> 20) & 255]
            total = regs[a] + b + (carry if op & (1 << 43) else 0)
            regs[dst] = total & MASK32
            if op & (1 << 47):
                carry = int(total > MASK32)
        elif prefix in (0xEED0000000000000 >> 51, 0xEED8000000000000 >> 51):
            assert (op >> 48) & 7 == 4 and op & (1 << 45)
            address = regs[a] | (regs[a + 1] << 32)
            if prefix == 0xEED0000000000000 >> 51:
                regs[dst] = memory[address]
            else:
                memory[address] = regs[dst]
        else:
            raise AssertionError(f'unknown original fixture instruction {op:016x}')
    raise AssertionError('fixture has no EXIT')
