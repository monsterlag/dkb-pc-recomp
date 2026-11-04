"""Original Maxwell GS emission/store fixtures; no captured program bytes."""
import struct

PT = 7 << 16
SCHED = 0x001f8000fc0007e0
EXIT = 0xe30000000000000f | PT


def movi(register, bits):
    return 0x0100000000000000 | (bits << 20) | (15 << 12) | PT | register


def movc(register, bank, word):
    return 0x4c98000000000000 | (15 << 39) | (bank << 34) | (word << 20) | PT | register


def ast(first, address, words=4):
    # InstAst: SrcB low byte; Ra/Rc=RZ; Imm11 bytes; AlSize=words-1.
    return (0xeff0000000000000 | (255 << 39) | ((words - 1) << 47) |
            (address << 20) | PT | (255 << 8) | first)


def ald(first, address, vertex, words=4):
    # InstAld: explicit primitive-vertex register SrcB; Ra=RZ, no physical indexing.
    return (0xefd8000000000000 | (vertex << 39) | ((words - 1) << 47) |
            (address << 20) | PT | (255 << 8) | first)


def out_vertex(cut=False):
    return 0xf6e0000000000000 | ((1 + 2 * cut) << 39) | PT | (255 << 8) | 255


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def geometry_header(output, invocations, max_vertices):
    header = [0] * 20
    header[0] = 1 | (3 << 5) | (4 << 10)
    header[2] = invocations << 24
    header[3] = {'points': 1, 'line_strip': 6, 'triangle_strip': 7}[output] << 24
    header[4] = max_vertices
    return struct.pack('<20I', *header)


def strip_program(output):
    """Emit two complete strips per invocation, with distinct literal vertices.

    Four vertices/2 invocations for lines; six vertices/3 invocations for triangles.
    Each vertex rewrites Position and color before OUT: outputs become undefined
    after EmitVertex and must not be assumed to retain earlier values.
    """
    triangle = output == 'triangle_strip'
    count = 3 if triangle else 2
    instructions = []
    for primitive in range(2):
        for vertex in range(count):
            position = ((-0.75, -0.5, 0, 1), (0.75, -0.5, 0, 1), (0, 0.5, 0, 1))[vertex]
            position = (position[0], position[1] + primitive * 0.25, position[2], position[3])
            color = (0.25 * (vertex + 1), 0.25 * (primitive + 1), 0.5, 1)
            instructions += [movi(i, bits(value)) for i, value in enumerate(position + color)]
            instructions += [ast(0, 0x70), ast(4, 0x80), out_vertex(cut=vertex == count - 1)]
    return geometry_header(output, 3 if triangle else 2, count * 2) + framed(instructions + [EXIT])


def input_program(output, vertices):
    """Copy selected input Position and Location6 to Position and Location2.

    Every authored vertex index must be valid for the request's input topology.
    The caller chooses the non-adjacent vertices for an adjacency topology.
    """
    instructions = []
    for index, vertex in enumerate(vertices):
        instructions += [movi(12, vertex), ald(0, 0x70, 12), ald(4, 0xe0, 12),
                         ast(0, 0x70), ast(4, 0xa0)]
        if output == 'points':
            instructions += [movi(8, bits(1)), ast(8, 0x6c, 1)]
        instructions += [out_vertex(cut=index == len(vertices) - 1)]
    return geometry_header(output, 2, len(vertices)) + framed(instructions + [EXIT])


def framed(instructions):
    words = []
    for i in range(0, len(instructions), 3):
        batch = instructions[i:i + 3]
        words += [SCHED] + batch + [0] * (3 - len(batch))
    return struct.pack('<' + 'Q' * len(words), *words)


def program(store=False, passthrough=False, stage='gs'):
    """Write a red point at Position(0,0,0,1), optionally store through user c3.

    Passthrough is an explicit negative contract control: no OUT is authored.
    It is not a substitute for the downstream varying-linkage protocol.
    """
    header = [0] * 20
    header[0] = 1 | (3 << 5) | ({'vs': 1, 'gs': 4, 'ps': 5}[stage] << 10)
    header[0] |= (int(store) << 16) | (int(passthrough) << 24)
    header[2] = 1 << 24
    header[3] = (7 if passthrough else 1) << 24
    header[4] = 1
    instructions = []
    if store:
        instructions += [movi(8, 0x1234abcd), movc(10, 3, 0), movc(11, 3, 1),
                         0xeed8000000000000 | (4 << 48) | (1 << 45) | (10 << 8) | PT | 8]
    if not passthrough:
        if stage == 'ps':
            header[18] = 15
            instructions += [movi(i, bits) for i, bits in enumerate((0x3f800000, 0, 0, 0x3f800000))]
        else:
            instructions += [movi(i, bits) for i, bits in enumerate((0, 0, 0, 0x3f800000, 0x3f800000, 0, 0, 0x3f800000))]
            instructions += [ast(0, 0x70), ast(3, 0x6c, 1), ast(4, 0x80)]
            if stage == 'gs':
                instructions += [0xf6e0000000000000 | (3 << 39) | PT | (255 << 8) | 255]
    return struct.pack('<20I', *header) + framed(instructions + [EXIT])


def instructions(data):
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    assert words[0] == 0x07230203
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 65535
        assert count and offset + count <= len(words)
        yield opcode, words[offset + 1:offset + count]
        offset += count
