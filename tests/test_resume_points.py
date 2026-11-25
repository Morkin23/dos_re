"""Waiting across a replay boundary: instruction deadline + declared resume points.

A toy real-mode program waits in a poll loop for a counter that only the INT 08h
handler increments, and the handler is delivered only at a frame boundary -- the
shape of a palette fade that waits for the next timer tick.  An authored adapter
for the waiting function parks at the published deadline on one of its declared
resume points with the interpreter's exact state, the boundary delivers the
interrupt, and the resume-point hook continues.  The interpreted original is the
oracle for every boundary and for the return.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from dos_re.cpu import CPU8086, CPUState
from dos_re.execution import (
    INTERPRETED_CPU_CARRIER,
    BackendAdapter,
    DependencyCapability,
    ImplementationCatalog,
    ImplementationDescriptor,
    ImplementationEntry,
    ImplementationOrigin,
    OverrideCategory,
    ProgramCoverage,
    bind_plan_implementations,
    plan_execution,
    profile_configuration,
    real_mode_point,
)
from dos_re.identity import (
    ExecutionPointIdentity,
    FunctionIdentity,
    ImageIdentity,
    ProgramIdentity,
    real_mode_address,
)
from dos_re.memory import Memory
from dos_re.player import GameFrontend, run_frame_steps
from dos_re.replay import GUEST_INSTRUCTION_COORDINATE, ReplayError, ReplayPoint, ReplayPointCoordinate
from dos_re.x86 import HookParkError

ROOT_DIR = __file__
IMAGE = ImageIdentity(ProgramIdentity("toy"), "TOY.EXE", "sha256", "0" * 64)
WAIT_CS, CALLER_CS, ISR_CS, DS = 0x1000, 0x2000, 0x3000, 0x0500
TICK, SNAP, DONE = 0x0100, 0x0102, 0x0104

# 1000:0000 the waiting far proc
WAIT_CODE = bytes.fromhex(
    "a10001"        # 0000 mov ax,[0100]      tick counter, read once
    "a30201"        # 0003 mov [0102],ax
    "b90001"        # 0006 mov cx,0100h
    "a10001"        # 0009 mov ax,[0100]      poll loop
    "3b060201"      # 000C cmp ax,[0102]
    "7502"          # 0010 jnz 0014
    "e2f5"          # 0012 loop 0009
    "ff060401"      # 0014 inc word [0104]
    "cb")           # 0018 retf
LOOP_IPS = (0x0009, 0x000C, 0x0010, 0x0012)
CALLER_CODE = bytes.fromhex("9a00000010" "ebfe")   # call far 1000:0000 / jmp $
ISR_CODE = bytes.fromhex("ff060001" "cf")         # inc word [0100] / iret


def _function(cs, ip):
    return str(FunctionIdentity(IMAGE, "real-mode", real_mode_address(cs, ip)))


def _point(cs, ip):
    return str(ExecutionPointIdentity(IMAGE, "real-mode", real_mode_address(cs, ip)))


WAIT_TARGET = _function(WAIT_CS, 0)
RESUME_POINTS = frozenset(_point(WAIT_CS, ip) for ip in LOOP_IPS)


def _machine() -> SimpleNamespace:
    mem = Memory()
    for cs, code in ((WAIT_CS, WAIT_CODE), (CALLER_CS, CALLER_CODE), (ISR_CS, ISR_CODE)):
        for i, b in enumerate(code):
            mem.wb(cs, i, b)
    mem.ww(0, 0x08 * 4, 0x0000)
    mem.ww(0, 0x08 * 4 + 2, ISR_CS)
    mem.ww(DS, TICK, 7)
    s = CPUState()
    s.cs, s.ip, s.ds, s.ss, s.sp = CALLER_CS, 0, DS, 0x0900, 0x0200
    s.flags = 0x0202
    return SimpleNamespace(cpu=CPU8086(mem, s), dos=None)


# ----------------------------------------------------------------- the adapter
def _wait_body(cpu, ip: int) -> None:
    """Authored body of the poll loop from loop position ``ip`` (the state of
    the original there is in the CPU); stops at the deadline or returns."""
    s, mem = cpu.s, cpu.mem
    deadline = cpu.instruction_deadline
    while True:
        if deadline is not None and cpu.instruction_count >= deadline:
            s.ip = ip                                   # park (validated by the CPU)
            return
        if ip == 0x0009:
            s.ax = mem.rw(s.ds, TICK)
            ip = 0x000C
        elif ip == 0x000C:
            cpu.set_sub_flags(s.ax, mem.rw(s.ds, SNAP), s.ax - mem.rw(s.ds, SNAP), 16)
            ip = 0x0010
        elif ip == 0x0010:
            ip = 0x0014 if s.ax != mem.rw(s.ds, SNAP) else 0x0012
        elif ip == 0x0012:
            s.cx = (s.cx - 1) & 0xFFFF
            ip = 0x0009 if s.cx else 0x0014
        else:                                           # 0014 inc / 0018 retf
            old = mem.rw(s.ds, DONE)
            mem.ww(s.ds, DONE, (old + 1) & 0xFFFF)
            cpu.set_incdec_flags(old, (old + 1) & 0xFFFF, 16, dec=False)
            s.ip = mem.rw(s.ss, s.sp)
            s.cs = mem.rw(s.ss, (s.sp + 2) & 0xFFFF)
            s.sp = (s.sp + 4) & 0xFFFF
            cpu.call_depth = max(0, cpu.call_depth - 1)
            cpu.instruction_count += 2
            return
        cpu.instruction_count += 1


def wait_entry(cpu) -> None:
    s, mem = cpu.s, cpu.mem
    s.ax = mem.rw(s.ds, TICK)
    mem.ww(s.ds, SNAP, s.ax)
    s.cx = 0x0100
    cpu.instruction_count += 3
    _wait_body(cpu, 0x0009)


def wait_resume(cpu) -> None:
    _wait_body(cpu, cpu.s.ip)


wait_entry.owns_time = True
wait_resume.owns_time = True


def _activate(runtime, targets) -> None:
    assert targets == (WAIT_TARGET,)
    hooks = runtime.cpu.replacement_hooks
    hooks[(WAIT_CS, 0)] = wait_entry
    for ip in LOOP_IPS:
        hooks[(WAIT_CS, ip)] = wait_resume


def _plan(resume_points=RESUME_POINTS, activate=_activate):
    root = _function(CALLER_CS, 0)
    coverage = ProgramCoverage(roots=(root,), reachable=frozenset({root, WAIT_TARGET}),
                               evidence_identity="toy-v1")
    baseline = ImplementationDescriptor(
        "interpreted", frozenset({root, WAIT_TARGET}), ImplementationOrigin.INTERPRETED,
        required_capabilities=frozenset({DependencyCapability.ORIGINAL_EXE.value,
                                         DependencyCapability.INTERPRETER.value}),
        implementation_digest="i1")
    authored = ImplementationDescriptor(
        "toy.wait", frozenset({WAIT_TARGET}), ImplementationOrigin.AUTHORED,
        category=OverrideCategory.FAITHFUL, properties=frozenset({"cpu-adapted"}),
        implementation_digest="w1", resume_points=resume_points)
    catalog = ImplementationCatalog((
        ImplementationEntry(baseline),
        ImplementationEntry(authored, implementation=wait_entry, adapters=(
            BackendAdapter("toy.wait/cpu", INTERPRETED_CPU_CARRIER, activate, "a1"),)),
    ))
    config = profile_configuration("development", program_identity="toy",
                                   selected_overrides=("toy.wait",),
                                   provider_preference=("toy.wait", "interpreted"))
    return plan_execution(config, coverage, catalog)


def _bound_machine():
    rt = _machine()
    bind_plan_implementations(rt, _plan(), carrier_id=INTERPRETED_CPU_CARRIER)
    return rt


def _state(rt):
    cpu = rt.cpu
    return (dataclasses.astuple(cpu.s), cpu.instruction_count, cpu.call_depth, bytes(cpu.mem.data))


def _replay(rt, targets):
    fe = GameFrontend(ROOT_DIR)
    args = SimpleNamespace(timer_irqs_per_frame=1)
    states = []
    for n, target in enumerate(targets):
        fe.advance_replay_frame(rt, args, n, ReplayPointCoordinate(
            ReplayPoint(n + 1, "toy"), GUEST_INSTRUCTION_COORDINATE, target))
        states.append(_state(rt))
    return states


# ------------------------------------------------------------------- the tests
def test_descriptor_validates_resume_points():
    with pytest.raises(ValueError, match="execution point identity"):
        ImplementationDescriptor("x", frozenset({WAIT_TARGET}), ImplementationOrigin.AUTHORED,
                                 category=OverrideCategory.FAITHFUL,
                                 resume_points=frozenset({_function(WAIT_CS, 9)}))
    other = ImageIdentity(ProgramIdentity("other"), "O.EXE", "sha256", "1" * 64)
    with pytest.raises(ValueError, match="image/address space"):
        ImplementationDescriptor("x", frozenset({WAIT_TARGET}), ImplementationOrigin.AUTHORED,
                                 category=OverrideCategory.FAITHFUL, resume_points=frozenset({
                                     str(ExecutionPointIdentity(other, "real-mode", "1000:0009"))}))
    with pytest.raises(ValueError, match="only authored"):
        ImplementationDescriptor("x", frozenset({WAIT_TARGET}), ImplementationOrigin.GENERATED,
                                 resume_points=RESUME_POINTS)
    assert real_mode_point(_point(WAIT_CS, 0x12)) == (WAIT_CS, 0x12)


def test_plan_digest_covers_resume_points_and_rejects_own_entry():
    assert _plan().plan_digest != _plan(resume_points=frozenset()).plan_digest
    with pytest.raises(ValueError, match="own entry targets"):
        _plan(resume_points=frozenset({_point(WAIT_CS, 0)}))


def test_binding_requires_a_hook_at_every_resume_point():
    def forgetful(runtime, targets):
        runtime.cpu.replacement_hooks[(WAIT_CS, 0)] = wait_entry

    with pytest.raises(RuntimeError, match="did not install a hook"):
        bind_plan_implementations(_machine(), _plan(activate=forgetful),
                                  carrier_id=INTERPRETED_CPU_CARRIER)
    rt = _bound_machine()
    assert rt.cpu.hook_resume_owner == {(WAIT_CS, ip): "toy.wait" for ip in LOOP_IPS}
    assert rt.cpu.hook_owner[(WAIT_CS, 0)] == "toy.wait"


@pytest.mark.parametrize("phase", range(4))
def test_parked_adapter_matches_interpreter_at_every_boundary(phase):
    # frame 1: ISR (2) + call (1) + prologue (3) + 5 loop iterations + ``phase``
    # -> the boundary lands in the poll loop at loop offset ``phase``; the INT
    # 08h delivered there changes the tick, so the resumed loop exits and the
    # function returns during frame 2.
    targets = [2 + 1 + 3 + 4 * 5 + phase, 2 + 1 + 3 + 4 * 5 + phase + 40, 400]
    oracle = _replay(_machine(), targets)
    candidate = _replay(_bound_machine(), targets)
    for n, (o, c) in enumerate(zip(oracle, candidate)):
        assert o == c, f"boundary {n + 1}"
    rt = _bound_machine()
    _replay(rt, targets[:1])
    assert (rt.cpu.s.cs, rt.cpu.s.ip) == (WAIT_CS, LOOP_IPS[phase])
    assert rt.cpu.mem.rw(DS, DONE) == 0 and rt.cpu.s.cx == 0x0100 - 5
    _replay(rt, targets[1:])
    assert rt.cpu.mem.rw(DS, DONE) == 1 and rt.cpu.s.cs == CALLER_CS


def test_without_resume_points_the_atomic_hook_crosses_the_point():
    rt = _machine()
    fe = GameFrontend(ROOT_DIR)
    args = SimpleNamespace(timer_irqs_per_frame=1)

    def atomic(cpu):
        deadline, cpu.instruction_deadline = cpu.instruction_deadline, None
        try:
            wait_entry(cpu)
        finally:
            cpu.instruction_deadline = deadline
    atomic.owns_time = True
    rt.cpu.replacement_hooks[(WAIT_CS, 0)] = atomic
    with pytest.raises(ReplayError, match="crossed replay point"):
        fe.advance_replay_frame(rt, args, 0, ReplayPointCoordinate(
            ReplayPoint(1, "toy"), GUEST_INSTRUCTION_COORDINATE, 30))


def test_park_outside_the_deadline_or_by_another_owner_fails_loud():
    rt = _bound_machine()
    rt.cpu.step()                                   # call far -> 1000:0000
    rt.cpu.instruction_deadline = None
    with pytest.raises(HookParkError, match="deadline None"):
        def early(cpu):
            cpu.s.ip = 0x0009
        early.owns_time = True
        rt.cpu.replacement_hooks[(WAIT_CS, 0)] = early
        rt.cpu.step()
    rt = _bound_machine()
    rt.cpu.step()
    rt.cpu.instruction_deadline = rt.cpu.instruction_count + 50
    with pytest.raises(HookParkError, match="park only at the published deadline"):
        rt.cpu.replacement_hooks[(WAIT_CS, 0)] = early
        rt.cpu.step()
    def foreign(cpu):
        cpu.s.cs, cpu.s.ip = WAIT_CS, 0x0009
    foreign.owns_time = True
    rt = _bound_machine()
    rt.cpu.replacement_hooks[(CALLER_CS, 0)] = foreign  # another implementation's hook
    rt.cpu.hook_owner[(CALLER_CS, 0)] = "someone.else"
    rt.cpu.instruction_deadline = 0
    with pytest.raises(HookParkError, match="owned by 'toy.wait'"):
        rt.cpu.step()


def test_live_frame_ends_early_when_a_hook_parks_and_resumes_next_frame():
    rt = _bound_machine()
    done = run_frame_steps(rt.cpu, 40)              # call + entry hook parks at count 40
    assert rt.cpu.instruction_count == 40 and done == 2
    assert (rt.cpu.s.cs, rt.cpu.s.ip) in {(WAIT_CS, ip) for ip in LOOP_IPS}
    assert rt.cpu.instruction_deadline is None and rt.cpu.parked is False
    from dos_re.interrupts import deliver_interrupt
    deliver_interrupt(rt, 0x08)
    run_frame_steps(rt.cpu, 40)
    assert rt.cpu.mem.rw(DS, DONE) == 1 and rt.cpu.s.cs == CALLER_CS


def test_run_frame_steps_is_plain_run_without_resume_points():
    rt = _machine()
    assert run_frame_steps(rt.cpu, 25) == 25
    assert rt.cpu.instruction_deadline is None


def test_unowned_stub_may_return_to_a_parked_resume_point():
    """A framework/BIOS stub (no plan owner) such as the default INT 08h
    handler's IRET legitimately returns into the parked point."""
    def iret_stub(cpu):
        cpu.s.cs, cpu.s.ip = WAIT_CS, 0x000C
    iret_stub.owns_time = True
    rt = _bound_machine()
    rt.cpu.replacement_hooks[(0xF000, 0xFF53)] = iret_stub
    rt.cpu.s.cs, rt.cpu.s.ip = 0xF000, 0xFF53
    rt.cpu.step()
    assert (rt.cpu.s.cs, rt.cpu.s.ip) == (WAIT_CS, 0x000C) and rt.cpu.parked is False
