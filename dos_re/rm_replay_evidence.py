"""Real-mode replay execution evidence for the Execution Atlas.

The real-mode producer half of the dos_re replay/Atlas contract, mirroring
:mod:`dos_re.pm_replay_evidence`.  The Atlas consumes ``FunctionVisit``,
``ObservedTransfer`` and ``ReplayExecutionEvidence`` (``dos_re.replay``); this
module makes a real-mode (CPU8086) oracle run produce them:

* :class:`RMFunctionObserver` -- entry/exit observation over the CPU's
  non-perturbing ``entry_probes`` seam.  Probes never replace execution, count
  instructions or re-poll IRQs, so the observed run is byte-identical to an
  unobserved one.
* :class:`RMCallTargetDiscovery` / :func:`discover_real_mode_call_targets` --
  for programs without Recovery IR: one replay pass over the existing
  ``coverage_telemetry`` seam that collects the targets of executed CALL
  instructions and the guest interrupt handlers that ran, i.e. the entry
  addresses the observer needs.
* :func:`observe_real_mode_replay` -- the oracle evidence pass: replay
  ``0..end`` once with observation and persist visits + transfers with
  ``ReplayArtifact.set_execution_evidence``.

Real-mode stepping (frame pacing, boundary rules, input application) belongs
to each frontend's :class:`~dos_re.replay.ReplayDriver` (the
``verification_drivers`` contract), so there is no generic real-mode driver
here: both passes take the frontend's oracle driver
(``GameFrontend.observation_driver``).  The driver must expose the restored
runtime as ``driver.runtime``.

Conventions (identical to the PM observer):

* Visits are frame-granular and COVERING: an entry during frame N records
  boundary N (the state before that frame); an exit records the boundary after
  the frame in which it was detected, so replaying [first_entry, last_exit]
  contains every recorded invocation.  Invocation counts and first entries
  are exact.
* Exits are detected exactly: at entry the observer records the return
  address and the SP the caller holds after the return, and probes that return
  address while the frame is live; reaching it with that SP closes the frame
  (and any frame above it).  Lazy stack unwinding is the backstop for returns
  that bypass their return address: frames whose SP has been unwound past are
  popped at the next entry on that stack and at :meth:`finish`.  A jump back
  to the entry of the live call (loop head) is therefore not an invocation;
  an interrupt-handler entry always is (an interrupt arriving again before the
  previous handler's return address executed closes the previous frame).
* A probed address counts as a call only if the stack top is the return
  address of a CALL to it (near ``E8``/``FF /2``, far ``9A``/``FF /3``, which
  includes ``push cs; call near``).  A jump to an entry is not an invocation.
* Guest interrupt handlers (hardware IRQs and ``INT n`` into guest code) are
  separate entries: their invocations are visits too, and the interrupted
  function -> handler edge is a transfer of kind ``interrupt``.
* Stacks are keyed by SS; an interrupt handler runs below the interrupted
  frames on the same stack, so it cannot satisfy their exit condition.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .replay import (
    ReplayArtifact,
    ReplayDriver,
    ReplayEvidenceRecorder,
    ReplayPoint,
)

Address = tuple[int, int]

#: Interrupt vectors whose guest handlers discovery tracks: hardware IRQs
#: (master 08h-0Fh, slave 70h-77h) plus every software vector a program may
#: hook -- i.e. all 256; only those pointing into the program range count.
_ALL_VECTORS = range(256)
_SEG_PREFIXES = (0x26, 0x2E, 0x36, 0x3E)
_REP_PREFIXES = (0xF2, 0xF3)
_MAX_SHADOW_FRAMES = 100_000
_CONVENTIONAL_TOP = 0xA0000

# discovery instruction classes (cached per address)
_OTHER, _CALL, _INT = 0, 1, 2


def _linear(cs: int, ip: int) -> int:
    return ((cs & 0xFFFF) << 4) + (ip & 0xFFFF)


def _rb(mem, seg: int, off: int) -> int | None:
    """Byte read restricted to conventional memory (no device side effects)."""
    lin = (((seg & 0xFFFF) << 4) + (off & 0xFFFF)) & 0xFFFFF
    if lin >= _CONVENTIONAL_TOP:
        return None
    return mem.data[lin]


def _rw(mem, seg: int, off: int) -> int | None:
    lo, hi = _rb(mem, seg, off), _rb(mem, seg, off + 1)
    if lo is None or hi is None:
        return None
    return lo | (hi << 8)


def program_range(runtime) -> tuple[int, int]:
    """Default program extent: from the load segment to the top of conventional memory."""
    program = getattr(runtime, "program", None)
    load = getattr(program, "load_segment", None)
    if load is None:
        raise ValueError("runtime has no program.load_segment; pass an explicit range")
    return ((load & 0xFFFF) << 4, _CONVENTIONAL_TOP)


# --------------------------------------------------------------------------
# observer
# --------------------------------------------------------------------------
class RMFunctionObserver:
    """Entry/exit + transfer observation over ``cpu.entry_probes`` (real mode).

    ``functions`` maps entry addresses ``(cs, ip)`` to stable function identity
    strings; ``interrupt_entries`` maps guest interrupt-handler entries to
    identity strings (a handler may also be called as a function: one entry
    can be in both maps).  ``point`` supplies the CURRENT stable replay point.
    """

    def __init__(
        self,
        cpu,
        functions: Mapping[Address, str],
        recorder: ReplayEvidenceRecorder,
        point: Callable[[], ReplayPoint],
        *,
        interrupt_entries: Mapping[Address, str] | None = None,
    ) -> None:
        self.cpu = cpu
        self.functions = {(int(c) & 0xFFFF, int(i) & 0xFFFF): str(v)
                          for (c, i), v in functions.items()}
        self.interrupts = {(int(c) & 0xFFFF, int(i) & 0xFFFF): str(v)
                           for (c, i), v in (interrupt_entries or {}).items()}
        self.recorder = recorder
        self.point = point
        #: active frames per stack segment: {ss: [(function_id, sp_after, ret), ...]}
        self._stacks: dict[int, list[tuple[str, int, Address]]] = {}
        #: return addresses of live frames -> number of live frames returning there
        self._ret_watch: Counter = Counter()
        if cpu.entry_probes is None:
            cpu.entry_probes = {}
        self._probes = cpu.entry_probes
        for addr in set(self.functions) | set(self.interrupts):
            existing = self._probes.get(addr)
            if existing is not None:
                raise RuntimeError(f"entry probe collision at {addr[0]:04X}:{addr[1]:04X}: {existing!r}")
            self._probes[addr] = self._fire
        self._owned = set(self._probes_owned())

    def _probes_owned(self):
        return [a for a, p in self._probes.items() if p == self._fire]

    # -- call shape ------------------------------------------------------------
    @staticmethod
    def call_kind(cpu, cs: int, ip: int) -> int:
        """Bytes the matching return pops (2 near, 4 far) if the stack top is
        the return address of a CALL to ``cs:ip``; 0 otherwise."""
        mem, s = cpu.mem, cpu.s
        ss, sp = s.ss & 0xFFFF, s.sp & 0xFFFF
        ret_ip = _rw(mem, ss, sp)
        if ret_ip is None:
            return 0
        # near E8 rel16 in the callee's segment (also `push cs; call near`)
        if _rb(mem, cs, ret_ip - 3) == 0xE8:
            rel = _rw(mem, cs, ret_ip - 2)
            if rel is not None and (ret_ip + rel) & 0xFFFF == ip:
                return 2
        ret_cs = _rw(mem, ss, sp + 2)
        # far 9A off seg
        if ret_cs is not None and _rb(mem, ret_cs, ret_ip - 5) == 0x9A \
                and _rw(mem, ret_cs, ret_ip - 4) == ip and _rw(mem, ret_cs, ret_ip - 2) == cs:
            return 4
        # near indirect FF /2 (2..4 bytes, +1 with a segment prefix)
        for length in (2, 3, 4, 5):
            if _rb(mem, cs, ret_ip - length) == 0xFF:
                modrm = _rb(mem, cs, ret_ip - length + 1)
                if modrm is not None and (modrm >> 3) & 7 == 2:
                    return 2
        # far indirect FF /3
        if ret_cs is not None:
            for length in (2, 3, 4, 5):
                if _rb(mem, ret_cs, ret_ip - length) == 0xFF:
                    modrm = _rb(mem, ret_cs, ret_ip - length + 1)
                    if modrm is not None and (modrm >> 3) & 7 == 3:
                        return 4
        return 0

    # -- frame bookkeeping -------------------------------------------------------
    def _push(self, stack: list, frame: tuple[str, int, Address]) -> None:
        stack.append(frame)
        ret = frame[2]
        self._ret_watch[ret] += 1
        if ret not in self._probes:
            self._probes[ret] = self._fire
        if len(stack) > _MAX_SHADOW_FRAMES:       # fail loud, never silently balloon
            raise RuntimeError("observer shadow stack overflow -- stack tracking has "
                               "diverged from the program's real call structure")

    def _pop(self, stack: list, exit_point: ReplayPoint) -> None:
        fid, _sp_after, ret = stack.pop()
        self.recorder.exit(fid, exit_point)
        left = self._ret_watch[ret] - 1
        if left:
            self._ret_watch[ret] = left
        else:
            del self._ret_watch[ret]
            if ret not in self._owned and self._probes.get(ret) == self._fire:
                del self._probes[ret]

    def _unwind(self, stack: list, level: int, exit_point: ReplayPoint) -> None:
        """Pop every frame whose post-return SP lies at or below ``level``."""
        while stack:
            sp_after = stack[-1][1]
            if ((level - sp_after) & 0xFFFF) >= 0x8000:
                break                       # level is below the frame: still live
            self._pop(stack, exit_point)

    def _fire(self, cpu) -> None:
        s = cpu.s
        addr = (s.cs & 0xFFFF, s.ip & 0xFFFF)
        sp = s.sp & 0xFFFF
        stack = self._stacks.setdefault(s.ss & 0xFFFF, [])
        here = self.point()
        after = ReplayPoint(here.ordinal + 1, here.timeline_id)
        if addr in self._ret_watch:
            # Exact return detection: execution reached the return address of a
            # live frame with the caller's SP.  Frames above it returned too
            # (or were abandoned); frames below stay live.
            for depth in range(len(stack) - 1, -1, -1):
                _fid, sp_after, ret = stack[depth]
                if ret == addr and sp_after == sp:
                    while len(stack) > depth:
                        self._pop(stack, after)
                    break
        if addr in self.functions:
            pops = self.call_kind(cpu, *addr)
            if pops:
                ret = (addr[0], _rw(cpu.mem, s.ss, sp)) if pops == 2 else \
                      (_rw(cpu.mem, s.ss, sp + 2), _rw(cpu.mem, s.ss, sp))
                sp_after = (sp + pops) & 0xFFFF
                top = stack[-1] if stack else None
                if top is not None and top[0] == self.functions[addr] \
                        and top[1] == sp_after and top[2] == ret:
                    return                  # jump back to the entry of the live call (loop head)
                self._enter(stack, self.functions[addr], "call", sp_after, ret, here, after)
                return
        if addr in self.interrupts:
            ret = (_rw(cpu.mem, s.ss, sp + 2), _rw(cpu.mem, s.ss, sp))
            sp_after = (sp + 6) & 0xFFFF
            top = stack[-1] if stack else None
            if top is not None and top[1] == sp_after and top[2] == ret:
                # The previous handler (or call) at this level returned with
                # IRET and the next interrupt arrived before the return address
                # executed again -- e.g. a guest blocked in a BIOS wait, whose
                # IP stays on the INT instruction while frames advance.  A
                # hardware interrupt is always a new invocation: close the old
                # frame (covering exit) and enter again.
                self._pop(stack, after)
            self._enter(stack, self.interrupts[addr], "interrupt", sp_after, ret, here, after)

    def _enter(self, stack, fid, kind, sp_after, ret, here, after) -> None:
        # Lazy backstop: frames whose post-return SP has been unwound past
        # (a return that bypassed its return address) close at the caller's level.
        self._unwind(stack, sp_after, after)
        caller = stack[-1][0] if stack else None
        self.recorder.enter(fid, here)
        if caller is not None:
            self.recorder.observe_transfer(caller, fid, kind, here)
        self._push(stack, (fid, sp_after, ret))

    def finish(self) -> None:
        """Close the books at a frame boundary: the current stack unwinds by
        its live SP; frames on other stacks have all returned (no handler is
        mid-flight at a boundary).  Calls still live stay ``incomplete``."""
        s = self.cpu.s
        here = self.point()
        current = s.ss & 0xFFFF
        for tag, stack in self._stacks.items():
            if tag == current:
                self._unwind(stack, s.sp & 0xFFFF, here)
            else:
                while stack:
                    self._pop(stack, here)

    def detach(self) -> None:
        for addr in self._probes_owned():
            del self._probes[addr]
        self._stacks.clear()
        self._ret_watch.clear()


# --------------------------------------------------------------------------
# discovery (programs without Recovery IR)
# --------------------------------------------------------------------------
@dataclass
class RMCallTargets:
    """Entry addresses observed in one replay pass."""

    calls: Counter = field(default_factory=Counter)            # target -> executed calls
    call_sites: Counter = field(default_factory=Counter)       # (site, target) -> count
    interrupt_entries: Counter = field(default_factory=Counter)  # handler -> entries
    interrupt_vectors: dict = field(default_factory=dict)      # handler -> sorted vectors
    outside_program: Counter = field(default_factory=Counter)  # skipped targets -> count
    instructions: int = 0

    def to_json(self) -> dict[str, Any]:
        def a(addr):
            return f"{addr[0]:04X}:{addr[1]:04X}"
        return {
            "instructions": self.instructions,
            "calls": {a(k): v for k, v in sorted(self.calls.items())},
            "call_sites": [{"site": a(s), "target": a(t), "count": n}
                           for (s, t), n in sorted(self.call_sites.items())],
            "interrupt_entries": {a(k): {"count": v, "vectors": self.interrupt_vectors.get(k, [])}
                                  for k, v in sorted(self.interrupt_entries.items())},
            "outside_program": {a(k): v for k, v in sorted(self.outside_program.items())},
        }


class RMCallTargetDiscovery:
    """``coverage_telemetry`` sink collecting executed CALL targets.

    Called by the CPU after each interpreted instruction with its start
    address; the opcode at that address is classified once (prefixes skipped)
    and, for a CALL, the CS:IP after it is the target.  ``INT n`` into guest
    code is an interrupt entry; hardware-IRQ handlers are found through
    :meth:`refresh_vectors` (the IVT targets inside the program range, checked
    against the executed addresses).  Read-only: never touches guest state.
    """

    def __init__(self, cpu, *, in_program: Callable[[int, int], bool]):
        self.cpu = cpu
        self.in_program = in_program
        self.result = RMCallTargets()
        self._kind: dict[Address, int] = {}
        self._vector_targets: dict[Address, set[int]] = {}

    def refresh_vectors(self) -> None:
        mem = self.cpu.mem
        targets: dict[Address, set[int]] = {}
        for vec in _ALL_VECTORS:
            off, seg = mem.rw(0, vec * 4), mem.rw(0, vec * 4 + 2)
            if (seg or off) and self.in_program(seg, off):
                targets.setdefault((seg, off), set()).add(vec)
        self._vector_targets = targets

    def _classify(self, key: Address) -> int:
        mem = self.cpu.mem
        cs, ip = key
        for k in range(6):
            op = _rb(mem, cs, ip + k)
            if op is None:
                return _OTHER
            if op in _SEG_PREFIXES or op in _REP_PREFIXES:
                continue
            if op in (0xE8, 0x9A):
                return _CALL
            if op == 0xFF:
                modrm = _rb(mem, cs, ip + k + 1)
                return _CALL if modrm is not None and (modrm >> 3) & 7 in (2, 3) else _OTHER
            if op == 0xCD:
                return _INT
            return _OTHER
        return _OTHER

    def record_interpreted_instruction(self, key: Address) -> None:
        r = self.result
        r.instructions += 1
        vt = self._vector_targets
        if vt and key in vt:
            # first instruction of a guest interrupt handler (hardware IRQ or INT n)
            r.interrupt_entries[key] += 1
            r.interrupt_vectors[key] = sorted(set(r.interrupt_vectors.get(key, [])) | vt[key])
        kind = self._kind.get(key)
        if kind is None:
            kind = self._kind[key] = self._classify(key)
        if kind == _OTHER:
            return
        s = self.cpu.s
        target = (s.cs & 0xFFFF, s.ip & 0xFFFF)
        if kind == _CALL:
            if self.in_program(*target):
                r.calls[target] += 1
                r.call_sites[(key, target)] += 1
            else:
                r.outside_program[target] += 1

    def record_hook_unverified(self, key, name) -> None:
        """Hooks are not instructions of the original; nothing to discover."""

    def bounded_original(self, key, reason) -> None:
        """Telemetry protocol method (dos_re.hooks); not relevant for discovery."""


def _ensure_oracle_profile(artifact: ReplayArtifact, driver: ReplayDriver) -> ReplayPoint:
    profile = driver.profile
    if profile.role != "oracle":
        raise ValueError("replay evidence requires an oracle profile")
    start = ReplayPoint(0, artifact.timeline_id)
    registered = {p.profile_id for p, _ in artifact.profiles()}
    if profile.profile_id not in registered:
        base = artifact.restore(artifact.capture_profile(), start)
        artifact.register_profile(profile, base_point=start, base_state=base)
    return start


def _end(artifact: ReplayArtifact, end: ReplayPoint | int | None) -> ReplayPoint:
    if end is None:
        raw = artifact.metadata.get("end_point")
        if not isinstance(raw, Mapping):
            raise ValueError("replay artifact has no recorded end point")
        return ReplayPoint.from_json(raw)
    if isinstance(end, int):
        return ReplayPoint(end, artifact.timeline_id)
    return end


def _runtime(driver):
    rt = getattr(driver, "runtime", None)
    if rt is None or getattr(rt, "cpu", None) is None:
        raise TypeError("the observation driver must expose the restored runtime as driver.runtime")
    return rt


def discover_real_mode_call_targets(
    artifact_path: str | Path,
    driver: ReplayDriver,
    *,
    end: ReplayPoint | int | None = None,
    program: tuple[int, int] | None = None,
    progress: Callable[[int], None] | None = None,
) -> RMCallTargets:
    """Replay ``0..end`` once on the oracle driver collecting entry addresses.

    ``program`` is the linear [start, stop) range of program code (default:
    :func:`program_range` of the restored runtime); targets outside it (BIOS
    stubs, framework vectors) are only counted.  The artifact is not modified
    beyond the oracle profile registration the driver itself may perform.
    """
    artifact = ReplayArtifact.open(artifact_path)
    start = _ensure_oracle_profile(artifact, driver)
    stop = _end(artifact, end)
    driver.restore(artifact.restore(driver.profile, start), start)
    rt = _runtime(driver)
    lo, hi = program if program is not None else program_range(rt)

    def in_program(cs: int, ip: int) -> bool:
        return lo <= _linear(cs, ip) < hi

    cpu = rt.cpu
    if cpu.coverage_telemetry is not None:
        raise RuntimeError("coverage_telemetry already attached; discovery needs it exclusively")
    sink = RMCallTargetDiscovery(cpu, in_program=in_program)
    cpu.coverage_telemetry = sink
    try:
        for ordinal in range(start.ordinal, stop.ordinal):
            sink.refresh_vectors()          # vectors change between frames (handlers installed)
            driver.replay_to(artifact, ReplayPoint(ordinal + 1, artifact.timeline_id))
            if progress is not None:
                progress(ordinal + 1)
    finally:
        cpu.coverage_telemetry = None
    return sink.result


def observe_real_mode_replay(
    artifact_path: str | Path,
    driver: ReplayDriver,
    functions: Mapping[Address, str],
    *,
    interrupt_entries: Mapping[Address, str] | None = None,
    end: ReplayPoint | int | None = None,
    provenance: Mapping[str, object] | None = None,
) -> ReplayEvidenceRecorder:
    """Oracle evidence pass: replay ``0..end`` once, observing function execution.

    Registers the oracle profile if needed, restores point 0, attaches an
    :class:`RMFunctionObserver` to the restored runtime, replays to ``end``
    (default: the artifact end; a shorter ``end`` observes only a validated
    prefix -- record why in ``provenance``), caches the end state when it is
    the artifact end, and persists visits + transfers with
    ``set_execution_evidence``.
    """
    artifact = ReplayArtifact.open(artifact_path)
    start = _ensure_oracle_profile(artifact, driver)
    stop = _end(artifact, end)
    driver.restore(artifact.restore(driver.profile, start), start)
    rt = _runtime(driver)
    recorder = ReplayEvidenceRecorder()
    observer = RMFunctionObserver(rt.cpu, functions, recorder, lambda: driver.current_point,
                                  interrupt_entries=interrupt_entries)
    try:
        driver.replay_to(artifact, stop)
        observer.finish()
    finally:
        observer.detach()
    artifact_end = _end(artifact, None)
    if stop == artifact_end and not artifact.has_cached(driver.profile, stop):
        artifact.cache(driver.profile, stop, driver.capture(),
                       metadata={"kind": "oracle-observation-end"})
    info = dict(provenance or {})
    info.setdefault("observer", "dos_re.rm_replay_evidence.RMFunctionObserver:v1")
    info.setdefault("observed_end", stop.ordinal)
    evidence = recorder.evidence(driver.profile, provenance=info)
    artifact.set_execution_evidence(driver.profile, evidence, visits=recorder.visits)
    return recorder
