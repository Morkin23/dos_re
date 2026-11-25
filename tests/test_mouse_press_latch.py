"""INT 33h press latch: a click between two AX=3 polls is not lost."""
from dos_re.dos import DOSMachine


class _S:
    ax = bx = cx = dx = 0


class _Cpu:
    def __init__(self):
        self.s = _S()


def _poll(dos):
    cpu = _Cpu()
    cpu.s.ax = 3
    dos.int33(cpu)
    return cpu.s.bx


def _machine():
    dos = DOSMachine.__new__(DOSMachine)
    dos.mouse_present = True
    dos.mouse_x, dos.mouse_y, dos.mouse_buttons = 320, 100, 0
    dos.mouse_press_latch = 0
    dos.mouse_range = [0, 639, 0, 199]
    return dos


def test_click_between_polls_is_reported_once():
    dos = _machine()
    dos.set_mouse_norm(0.5, 0.5, 1)
    dos.set_mouse_norm(0.5, 0.5, 0)
    assert _poll(dos) & 1 == 1
    assert _poll(dos) & 1 == 0


def test_held_button_stays_down_and_release_is_seen():
    dos = _machine()
    dos.set_mouse_norm(0.5, 0.5, 2)
    assert _poll(dos) == 2
    assert _poll(dos) == 2
    dos.set_mouse_norm(0.5, 0.5, 0)
    assert _poll(dos) == 0


def test_repeated_samples_of_a_held_button_do_not_relatch():
    dos = _machine()
    dos.set_mouse_norm(0.5, 0.5, 1)
    assert _poll(dos) == 1
    dos.set_mouse_norm(0.5, 0.5, 1)
    dos.set_mouse_norm(0.5, 0.5, 0)
    assert _poll(dos) == 0
