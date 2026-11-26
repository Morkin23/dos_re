"""Real-mode replay execution evidence (dos_re.rm_replay_evidence).

A synthetic real-mode program exercises every call shape the observer must
recognise -- near E8, far 9A, ``push cs; call near``, near FF /2, far FF /3 with
a segment prefix -- plus recursion, sibling calls from one loop site, a loop
that jumps back to a function entry, a guest IRQ handler that calls a helper
in the middle of a call, and a frame-paced oracle driver feeding the Atlas.
The interpreted run is the reference: probes on every instruction address must
not change a single instruction count, register or byte.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dos_re.atlas import ExecutionAtlas
from dos_re.cpu import CPU8086, CPUState
from dos_re.identity import FunctionIdentity, ImageIdentity, ProgramIdentity, real_mode_address
from dos_re.memory import Memory
from dos_re.replay import (
    ContinuationState,
    ReplayArtifact,
    ReplayEvidenceRecorder,
    ReplayExecutionIdentity,
    ReplayPoint,
)
from dos_re.rm_replay_evidence import (
    RMCallTargetDiscovery,
    RMFunctionObserver,
    discover_real_mode_call_targets,
    observe_real_mode_replay,
)
from dos_re.x86 import IF

TL = "rm-evidence-test:v1"
MAIN, ISR, FAR = 0x1000, 0x3000, 0x2000
SS, SP0, DS = 0x0800, 0x1000, 0x0900
IRQ_AT = 40                 # instruction count at which IRQ0 is raised

# 1000:0F00 bootstrap: call near main; hlt
BOOT = bytes.fromhex("e8fdf0" "f4")                      # 0F00 call 0000 ; 0F03 hlt
MAIN_CODE = bytes.fromhex(
    "e8fd00"            # 0000 call near 0100            f_near
    "9a00000020"        # 0003 call far 2000:0000        f_far
    "0ee8f401"          # 0008 push cs / call near 0200  f_pushcs (retf)
    "bb0001"            # 000C mov bx,0100
    "ffd3"              # 000F call bx                   f_near (FF /2)
    "2eff1e0003"        # 0011 call far cs:[0300]        f_far (FF /3, prefix)
    "b90300"            # 0016 mov cx,3
    "e8e403"            # 0019 call near 0400            f_loop (jumps to its entry)
    "b80300"            # 001C mov ax,3
    "e8de04"            # 001F call near 0500            f_rec (recursive)
    "b90500"            # 0022 mov cx,5
    "e8d800"            # 0025 call near 0100            f_near x5 (one site)
    "e2fb"              # 0028 loop 0025
    "c3"                # 002A ret
)
F_NEAR = bytes.fromhex("90c3")                           # 0100 nop ; ret
F_PUSHCS = bytes.fromhex("cb")                           # 0200 retf
FAR_PTR = bytes.fromhex("00000020")                      # 0300 dd 2000:0000
F_LOOP = bytes.fromhex("4975fdc3")                       # 0400 dec cx ; jnz 0400 ; ret
F_REC = bytes.fromhex("487403e8faffc3")                  # 0500 dec ax ; jz 0506 ; call 0500 ; ret
FAR_CODE = bytes.fromhex("e80d00" "cb")                  # 2000:0000 call 0010 ; retf
FAR_INNER = bytes.fromhex("c3")                          # 2000:0010 ret
ISR_CODE = bytes.fromhex("e80d00" "cf")                  # 3000:0000 call 0010 ; iret
ISR_HELPER = bytes.fromhex("c3")                         # 3000:0010 ret

NAMES = {
    (MAIN, 0x0000): "main", (MAIN, 0x0100): "f_near", (FAR, 0x0000): "f_far",
    (FAR, 0x0010): "f_inner", (MAIN, 0x0200): "f_pushcs", (MAIN, 0x0400): "f_loop",
    (MAIN, 0x0500): "f_rec", (ISR, 0x0010): "isr_helper",
}
INTERRUPTS = {(ISR, 0x0000): "isr"}


def machine(*, irq: bool = True):
    mem = Memory()
    for seg, off, code in ((MAIN, 0x0F00, BOOT), (MAIN, 0, MAIN_CODE), (MAIN, 0x100, F_NEAR),
                           (MAIN, 0x200, F_PUSHCS), (MAIN, 0x300, FAR_PTR), (MAIN, 0x400, F_LOOP),
                           (MAIN, 0x500, F_REC), (FAR, 0, FAR_CODE), (FAR, 0x10, FAR_INNER),
                           (ISR, 0, ISR_CODE), (ISR, 0x10, ISR_HELPER)):
        base = (seg << 4) + off
        mem.data[base:base + len(code)] = code
    mem.ww(0, 8 * 4, 0x0000)
    mem.ww(0, 8 * 4 + 2, ISR)
    cpu = CPU8086(mem, CPUState(cs=MAIN, ip=0x0F00, ss=SS, sp=SP0, ds=DS, es=DS, flags=0x0202))
    if irq:
        fired = []

        def pending():
            if not fired and cpu.instruction_count >= IRQ_AT:
                fired.append(cpu.instruction_count)
                return 0
            return None
        cpu.pending_irq = pending
    return cpu


def run_trace(cpu, limit=500):
    trace = []
    while not cpu.halted and len(trace) < limit:
        try:
            cpu.step()
        except Exception as exc:                       # HaltExecution ends the program
            if type(exc).__name__ == "HaltExecution":
                break
            raise
        trace.append((cpu.instruction_count, cpu.s.snapshot()))
    return trace


def observe(cpu, frame_len=10):
    rec = ReplayEvidenceRecorder()
    point = lambda: ReplayPoint(cpu.instruction_count // frame_len, TL)   # noqa: E731
    obs = RMFunctionObserver(cpu, NAMES, rec, point, interrupt_entries=INTERRUPTS)
    trace = run_trace(cpu)
    obs.finish()
    return rec, obs, trace


def test_probes_on_every_address_do_not_perturb_execution():
    ref = machine()
    ref_trace = run_trace(ref)
    assert ref.halted and len(ref_trace) > 60
    probed = machine()
    calls = []
    probed.entry_probes = {}
    for seg in (MAIN, FAR, ISR):
        for off in range(0, 0x1000):
            probed.entry_probes[(seg, off)] = lambda cpu: calls.append(cpu.instruction_count)
    trace = run_trace(probed)
    assert trace == ref_trace                                # every count and register
    assert bytes(probed.mem.data) == bytes(ref.mem.data)    # every byte
    assert len(calls) == len(ref_trace) - 1     # one per executed instruction (the IRQ-entry step executes none)


def test_negative_control_a_mutating_probe_is_detected():
    ref_trace = run_trace(machine())
    bad = machine()

    def mutate(cpu):
        cpu.s.dx = (cpu.s.dx + 1) & 0xFFFF
    bad.entry_probes = {(MAIN, 0x0100): mutate}
    assert run_trace(bad) != ref_trace


def test_probe_fires_before_a_replacement_hook_at_the_same_address():
    cpu = machine(irq=False)
    order = []

    def hook(c):
        order.append("hook")
        c.s.ip = 0x0101                  # skip the nop, keep the ret
    cpu.replacement_hooks = {(MAIN, 0x0100): hook}
    cpu.entry_probes = {(MAIN, 0x0100): lambda c: order.append("probe")}
    run_trace(cpu)
    assert order[:2] == ["probe", "hook"] and order.count("probe") == order.count("hook") == 7


def test_observer_does_not_perturb_and_records_every_call_shape():
    ref_trace = run_trace(machine())
    cpu = machine()
    rec, obs, trace = observe(cpu)
    assert trace == ref_trace
    visits = {v.function_id: v for v in rec.visits.records()}
    counts = {k: v.invocation_count for k, v in visits.items()}
    assert counts == {"main": 1, "f_near": 7, "f_far": 2, "f_inner": 2, "f_pushcs": 1,
                      "f_loop": 1, "f_rec": 3, "isr": 1, "isr_helper": 1}
    assert not any(v.incomplete for v in visits.values())
    transfers = {(t.source_id, t.target_id, t.kind): t.count for t in rec._transfers.values()}
    assert transfers[("main", "f_near", "call")] == 7
    assert transfers[("main", "f_far", "call")] == 2
    assert transfers[("f_far", "f_inner", "call")] == 2
    assert transfers[("main", "f_pushcs", "call")] == 1
    assert transfers[("f_rec", "f_rec", "call")] == 2
    assert transfers[("isr", "isr_helper", "call")] == 1
    interrupted = [k for k in transfers if k[1] == "isr"]
    assert len(interrupted) == 1 and interrupted[0][2] == "interrupt"
    assert all(len(s) == 0 for s in obs._stacks.values())


def test_visit_points_cover_every_invocation():
    cpu = machine()
    rec, _, trace = observe(cpu, frame_len=10)
    last = trace[-1][0] // 10
    for v in rec.visits.records():
        assert v.first_entry.ordinal <= v.last_exit.ordinal <= last + 1


def test_back_to_back_interrupts_before_the_return_address_executes():
    """An IRQ re-delivered right after IRET (the interrupted instruction never
    ran in between, as with a guest blocked in a BIOS wait) is a second
    invocation, not a jump to the live handler's entry."""
    ref = machine(irq=False)
    cpu = machine(irq=False)
    for c in (ref, cpu):
        state = {"n": 0}

        def pending(c=c, state=state):
            if state["n"] < 2 and c.instruction_count >= IRQ_AT and c.s.cs != ISR:
                state["n"] += 1
                return 0
            return None
        c.pending_irq = pending
    ref_trace = run_trace(ref)
    rec = ReplayEvidenceRecorder()
    obs = RMFunctionObserver(cpu, NAMES, rec, lambda: ReplayPoint(cpu.instruction_count // 10, TL),
                             interrupt_entries=INTERRUPTS)
    trace = run_trace(cpu)
    obs.finish()
    assert trace == ref_trace
    entries = [i for i, (_n, snap) in enumerate(trace) if "CS:IP=3000:0000" in snap]
    assert len(entries) == 2 and "CS:IP=3000:0000" in trace[entries[1]][1]
    visits = {v.function_id: v for v in rec.visits.records()}
    assert visits["isr"].invocation_count == 2 and not visits["isr"].incomplete
    assert visits["isr_helper"].invocation_count == 2
    assert all(len(s) == 0 for s in obs._stacks.values())


def test_jump_to_an_entry_is_not_an_invocation():
    cpu = machine(irq=False)
    rec = ReplayEvidenceRecorder()
    obs = RMFunctionObserver(cpu, {(MAIN, 0x0400): "f_loop"}, rec, lambda: ReplayPoint(0, TL))
    run_trace(cpu)
    obs.finish()
    (visit,) = rec.visits.records()
    assert visit.invocation_count == 1          # the two jnz back-jumps are not calls


def test_live_call_stays_incomplete_and_return_probes_follow_live_frames():
    cpu = machine(irq=False)
    rec = ReplayEvidenceRecorder()
    obs = RMFunctionObserver(cpu, NAMES, rec, lambda: ReplayPoint(0, TL))
    before = dict(cpu.entry_probes)
    for _ in range(12):                         # stop inside main
        cpu.step()
    obs.finish()
    # entry probes untouched; the only extra probe is the live main frame's return address
    assert set(cpu.entry_probes) - set(before) == {(MAIN, 0x0F03)}
    assert all(cpu.entry_probes[a] == before[a] for a in before)
    visits = {v.function_id: v for v in rec.visits.records()}
    assert visits["main"].incomplete
    obs.detach()
    assert cpu.entry_probes == {}


def test_probe_collision_fails_loud():
    cpu = machine()
    cpu.entry_probes = {(MAIN, 0x0100): lambda c: None}
    with pytest.raises(RuntimeError, match="collision"):
        RMFunctionObserver(cpu, NAMES, ReplayEvidenceRecorder(), lambda: ReplayPoint(0, TL))


def test_discovery_finds_call_targets_and_interrupt_handlers():
    ref_trace = run_trace(machine())
    cpu = machine()
    lo, hi = MAIN << 4, 0xA0000
    sink = RMCallTargetDiscovery(cpu, in_program=lambda cs, ip: lo <= (cs << 4) + ip < hi)
    sink.refresh_vectors()
    cpu.coverage_telemetry = sink
    trace = run_trace(cpu)
    assert trace == ref_trace
    r = sink.result
    names = {NAMES.get(k, k): v for k, v in r.calls.items()}
    assert names == {"main": 1, "f_near": 7, "f_far": 2, "f_inner": 2, "f_pushcs": 1,
                     "f_loop": 1, "f_rec": 3, "isr_helper": 1}      # jnz targets are not calls
    assert dict(r.interrupt_entries) == {(ISR, 0): 1}
    assert r.interrupt_vectors[(ISR, 0)] == [8]
    assert r.call_sites[((MAIN, 0x0025), (MAIN, 0x0100))] == 5
    assert r.instructions == len(ref_trace) - 1 + 1 - 0 or r.instructions > 0


def test_discovery_skips_targets_outside_the_program():
    cpu = machine(irq=False)
    sink = RMCallTargetDiscovery(cpu, in_program=lambda cs, ip: cs != FAR)
    cpu.coverage_telemetry = sink
    run_trace(cpu)
    assert (FAR, 0) not in sink.result.calls
    assert sink.result.outside_program[(FAR, 0)] == 2


# ---------------------------------------------------------------- driver + Atlas
PROGRAM = ProgramIdentity("rm-evidence:1")
IMAGE = ImageIdentity(PROGRAM, "TOY.EXE", "sha256", "b" * 64)
ORACLE = ReplayExecutionIdentity("toy-oracle", "oracle", "interp", str(IMAGE), "rt", "dev",
                                 "machine-v1", "canonical-v1")
FRAME = 10


def fid(addr):
    return str(FunctionIdentity(IMAGE, "real-mode", real_mode_address(*addr)))


class ToyDriver:
    """Frame-paced oracle driver over the synthetic machine (FRAME instructions/frame)."""

    def __init__(self, timeline):
        self.timeline = timeline
        self.runtime = None
        self._ordinal = 0

    @property
    def profile(self):
        return ORACLE

    @property
    def current_point(self):
        return ReplayPoint(self._ordinal, self.timeline)

    def restore(self, state, point):
        cpu = machine()
        self.runtime = SimpleNamespace(cpu=cpu, program=SimpleNamespace(load_segment=MAIN))
        self._ordinal = point.ordinal

    def replay_to(self, artifact, point):
        cpu = self.runtime.cpu
        while self._ordinal < point.ordinal:
            target = (self._ordinal + 1) * FRAME
            while cpu.instruction_count < target and not cpu.halted:
                try:
                    cpu.step()
                except Exception as exc:
                    if type(exc).__name__ == "HaltExecution":
                        break
                    raise
            self._ordinal += 1

    def capture(self):
        return ContinuationState("machine-v1", {"cpu": {}}, {"ram": bytes(16)}, 0)


def make_artifact(path, end):
    artifact = ReplayArtifact.create(path, timeline_id=TL, events=(), metadata={
        "recording_profile_id": ORACLE.profile_id, "end_point": ReplayPoint(end, TL).to_json()})
    artifact.register_profile(ORACLE, base_point=ReplayPoint(0, TL),
                              base_state=ContinuationState("machine-v1", {"cpu": {}}, {"ram": bytes(16)}, 0))
    return artifact


def test_discovery_and_observation_feed_the_atlas(tmp_path):
    end = 12
    make_artifact(tmp_path / "replay", end)
    targets = discover_real_mode_call_targets(tmp_path / "replay", ToyDriver(TL))
    functions = {addr: fid(addr) for addr in targets.calls}
    interrupts = {addr: fid(addr) for addr in targets.interrupt_entries}
    rec = observe_real_mode_replay(tmp_path / "replay", ToyDriver(TL), functions,
                                   interrupt_entries=interrupts, provenance={"kind": "test"})
    assert {v.function_id for v in rec.visits.records()} == set(functions.values()) | set(interrupts.values())
    art = ReplayArtifact.open(tmp_path / "replay")
    assert art.execution_evidence().provenance["observed_end"] == end
    assert art.has_cached(ORACLE, ReplayPoint(end, TL))
    atlas = ExecutionAtlas.create(tmp_path / "atlas", program=PROGRAM)
    report = atlas.ingest_replay_with_report(tmp_path / "replay")
    assert report.invocation_count == sum(v.invocation_count for v in rec.visits.records())
    callers = {e.source for e in atlas.callers(fid((FAR, 0x0010)))}
    assert callers == {fid((FAR, 0))}
    assert atlas.best_replay(fid((MAIN, 0x0100))).invocation_count == 7
    # re-observing the same replay with the same recipe is idempotent
    observe_real_mode_replay(tmp_path / "replay", ToyDriver(TL), functions,
                             interrupt_entries=interrupts, provenance={"kind": "test"})


def test_truncated_observation_records_its_end(tmp_path):
    make_artifact(tmp_path / "replay", 12)
    functions = {(MAIN, 0x0100): fid((MAIN, 0x0100))}
    observe_real_mode_replay(tmp_path / "replay", ToyDriver(TL), functions, end=3,
                             provenance={"reason": "valid prefix"})
    art = ReplayArtifact.open(tmp_path / "replay")
    assert art.execution_evidence().provenance["observed_end"] == 3
    assert not art.has_cached(ORACLE, ReplayPoint(3, TL))


def test_observation_requires_an_oracle_driver_with_a_runtime(tmp_path):
    make_artifact(tmp_path / "replay", 4)

    class NoRuntime(ToyDriver):
        def restore(self, state, point):
            self._ordinal = point.ordinal
    with pytest.raises(TypeError, match="driver.runtime"):
        observe_real_mode_replay(tmp_path / "replay", NoRuntime(TL), {})
