"""IVT[33h] mouse driver vector for programs that test for a driver."""
from dos_re.memory import Memory
from dos_re.runtime_core import (MOUSE_DRIVER_ENTRY, _init_bios_environment,
                                 install_mouse_driver_vector)


def _vector(mem):
    return mem.rw(0, 0x33 * 4 + 2), mem.rw(0, 0x33 * 4)


def test_power_on_environment_leaves_mouse_vector_null():
    mem = Memory()
    _init_bios_environment(mem)
    assert _vector(mem) == (0, 0)


def test_install_points_vector_at_int33_iret_stub():
    mem = Memory()
    _init_bios_environment(mem)
    assert install_mouse_driver_vector(mem) is True
    seg, off = _vector(mem)
    assert (seg, off) == MOUSE_DRIVER_ENTRY
    entry = seg * 16 + off
    # A detection test reads the first byte: it must not be IRET (CFh).
    assert bytes(mem.data[entry:entry + 3]) == b"\xCD\x33\xCF"


def test_install_keeps_a_program_owned_vector():
    mem = Memory()
    mem.data[0xCC:0xD0] = b"\x34\x12\x00\x20"
    assert install_mouse_driver_vector(mem) is False
    assert _vector(mem) == (0x2000, 0x1234)
