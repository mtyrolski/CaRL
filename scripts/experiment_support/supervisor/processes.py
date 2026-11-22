"""Linux process-group ownership, tracking and controlled termination."""

import ctypes
import errno
import os
import select
import signal
import subprocess
import time
from pathlib import Path


_LIBC = ctypes.CDLL(None, use_errno=True)


def _call_pidfd(name: str, *args: ctypes.c_int | ctypes.c_uint | ctypes.c_void_p) -> int:
    function = getattr(_LIBC, name, None)
    if function is None:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))
    result: int = function(*args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def pidfd_open(pid: int) -> int:
    """Open a process handle even on CPython builds with older Linux headers."""
    try:
        open_handle = os.pidfd_open
    except AttributeError:
        return _call_pidfd("pidfd_open", ctypes.c_int(pid), ctypes.c_uint(0))
    return open_handle(pid)


def pidfd_send_signal(fd: int, sig: int) -> None:
    """Send a signal through a process handle using CPython or libc."""
    try:
        send_signal = signal.pidfd_send_signal
    except AttributeError:
        _call_pidfd("pidfd_send_signal", ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))
    else:
        send_signal(fd, sig)


def boot_id() -> str:
    """Linux boot identifier; PID identities do not survive a reboot."""
    return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()


def proc_metadata(pid: int) -> tuple[int, int, int] | None:
    """Return process start ticks, process group and session, if accessible."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[19]), int(fields[2]), int(fields[3])
    except (OSError, IndexError, ValueError):
        return None


def proc_start(pid: int) -> int | None:
    metadata = proc_metadata(pid)
    return metadata[0] if metadata else None


def owned_process(stage: dict) -> bool:
    pid = stage.get("pid")
    start = stage.get("proc_start")
    if not isinstance(pid, int) or not isinstance(start, int):
        return False
    expected_boot = stage.get("boot_id")
    if expected_boot is not None and expected_boot != boot_id():
        return False
    return proc_metadata(pid) == (start, pid, pid)


def _pidfd_exited(fd: int) -> bool:
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    return bool(poller.poll(0))


def _open_member(pid: int, metadata: tuple[int, int, int]) -> int | None:
    """Open a process handle only if its identity is stable across the open."""
    if proc_metadata(pid) != metadata:
        return None
    try:
        fd = pidfd_open(pid)
    except OSError:
        return None
    if proc_metadata(pid) != metadata or _pidfd_exited(fd):
        os.close(fd)
        return None
    return fd


def _group_members(group: int) -> list[tuple[int, tuple[int, int, int]]]:
    members = []
    with os.scandir("/proc") as entries:
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            metadata = proc_metadata(pid)
            if metadata is not None and metadata[1:] == (group, group):
                members.append((pid, metadata))
    return members


def group_has_live_members(group: int) -> bool:
    """Fail closed when a registered process group still has a live member."""
    for pid, metadata in _group_members(group):
        try:
            fd = pidfd_open(pid)
        except OSError:
            if proc_metadata(pid) == metadata:
                return True
            continue
        try:
            if not _pidfd_exited(fd) and proc_metadata(pid) == metadata:
                return True
        finally:
            os.close(fd)
    return False


def track_group(stage: dict, handles: dict[int, tuple[int, int]]) -> int:
    """Retain pidfds observed while the registered group leader is alive."""
    if not owned_process(stage):
        return 0
    leader_pid = stage["pid"]
    start = stage["proc_start"]
    leader_entry = handles.get(leader_pid)
    if (
        leader_entry is None
        or leader_entry[1] != start
        or _pidfd_exited(leader_entry[0])
    ):
        if leader_entry is not None:
            os.close(leader_entry[0])
        leader_fd = _open_member(leader_pid, (start, leader_pid, leader_pid))
        if leader_fd is None:
            handles.pop(leader_pid, None)
            return 0
        handles[leader_pid] = leader_fd, start
    else:
        leader_fd = leader_entry[0]
    added = 0
    for pid, metadata in _group_members(leader_pid):
        if _pidfd_exited(leader_fd):
            break
        previous = handles.get(pid)
        if (
            previous is not None
            and previous[1] == metadata[0]
            and not _pidfd_exited(previous[0])
        ):
            continue
        fd = _open_member(pid, metadata)
        if fd is None:
            continue
        if _pidfd_exited(leader_fd):
            os.close(fd)
            break
        if previous is not None:
            os.close(previous[0])
        handles[pid] = fd, metadata[0]
        added += 1
    return added


def stop_tracked(handles: dict[int, tuple[int, int]]) -> bool:
    """Kill only previously verified processes, even if their leader has gone."""
    successful = True
    for fd, _ in handles.values():
        if not _pidfd_exited(fd):
            try:
                pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                successful = False
    confirm_at = time.monotonic() + 0.25
    while time.monotonic() < confirm_at and any(
        not _pidfd_exited(fd) for fd, _ in handles.values()
    ):
        time.sleep(0.01)
    return successful and all(_pidfd_exited(fd) for fd, _ in handles.values())


def close_tracked(handles: dict[int, tuple[int, int]]) -> None:
    for fd, _ in handles.values():
        os.close(fd)
    handles.clear()


def stop_owned(stage: dict, grace_seconds: float = 3.0) -> bool:
    """Stop a verified session using pidfds; never signal a recycled PID.

    A live leader anchors the scan. Group members are frozen before TERM so a
    disappearing leader cannot orphan an ignoring child. If the leader was
    already gone, ownership cannot safely be reconstructed from a bare PGID.
    """
    if not owned_process(stage):
        return False
    pid = stage["pid"]
    leader = _open_member(pid, (stage["proc_start"], pid, pid))
    if leader is None:
        return False
    handles = {pid: leader}
    frozen: set[int] = set()
    complete_scan = False
    inaccessible = False
    try:
        try:
            pidfd_send_signal(leader, signal.SIGSTOP)
        except OSError:
            return False
        frozen.add(leader)
        # A scan after the last newly frozen member is necessary: that member
        # could have forked between enumeration and SIGSTOP.
        for _ in range(32):
            if _pidfd_exited(leader):
                break
            added = 0
            for member_pid, metadata in _group_members(pid):
                if member_pid in handles or _pidfd_exited(leader):
                    continue
                fd = _open_member(member_pid, metadata)
                if fd is None:
                    inaccessible |= proc_metadata(member_pid) == metadata
                    continue
                if _pidfd_exited(leader):
                    os.close(fd)
                    break
                try:
                    pidfd_send_signal(fd, signal.SIGSTOP)
                except OSError as exc:
                    inaccessible |= not isinstance(exc, ProcessLookupError)
                    os.close(fd)
                    continue
                handles[member_pid] = fd
                frozen.add(fd)
                added += 1
            if not added:
                complete_scan = not _pidfd_exited(leader) and not inaccessible
                break
        for fd in frozen:
            try:
                pidfd_send_signal(fd, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError:
                complete_scan = False
        for fd in frozen:
            try:
                pidfd_send_signal(fd, signal.SIGCONT)
            except ProcessLookupError:
                pass
            except OSError:
                complete_scan = False
        frozen.clear()
        grace = max(0.0, grace_seconds)
        stop_at = time.monotonic() + grace
        stop_epoch = time.time() + grace
        while (
            time.monotonic() < stop_at
            and time.time() < stop_epoch
            and any(not _pidfd_exited(fd) for fd in handles.values())
        ):
            time.sleep(min(0.05, max(0.0, stop_at - time.monotonic())))
        for fd in handles.values():
            if not _pidfd_exited(fd):
                try:
                    pidfd_send_signal(fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    complete_scan = False
        # Allow the kernel a short interval to deliver SIGKILL. Do not claim a
        # complete stop for an uninterruptible process or an incomplete scan.
        confirm_at = time.monotonic() + 0.25
        while time.monotonic() < confirm_at and any(
            not _pidfd_exited(fd) for fd in handles.values()
        ):
            time.sleep(0.01)
        return complete_scan and all(_pidfd_exited(fd) for fd in handles.values())
    finally:
        # Never leave our group suspended after an unexpected exception.
        for fd in frozen:
            try:
                pidfd_send_signal(fd, signal.SIGCONT)
            except OSError:
                pass
        for fd in handles.values():
            os.close(fd)


def stage_leader(gate_fd: int, command: list[str]) -> int:
    """Keep the registered session alive until its adopted descendants exit."""
    try:
        permitted = os.read(gate_fd, 1) == b"1"
    finally:
        os.close(gate_fd)
    if not permitted:
        return 125
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A stage command is required")
    # Linux subreapers adopt orphaned descendants, including double-forks.
    if _LIBC.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))

    def keep_leader_alive(_signum: int, _frame: object) -> None:
        # The watchdog signals every captured member; this process remains to
        # retain ownership and adopt children while they shut down.
        return None

    signal.signal(signal.SIGTERM, keep_leader_alive)
    child = subprocess.Popen(command, close_fds=True)
    code = child.wait()
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break
        except InterruptedError:
            continue
    return code if code >= 0 else 128 - code
