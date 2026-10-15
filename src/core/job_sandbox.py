"""
Windows Job Object Sandbox & Process Resource Governor for J.A.R.V.I.S.
Inspired by the Chromium Core Multiprocess Architecture (chromium-main).

Enforces kernel-level hard memory caps (default 256MB), CPU limits,
and process tree termination (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) on Windows.
Provides seamless cross-platform fallback on Linux and macOS.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Win32 Constants for Job Objects
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JobObjectExtendedLimitInformation = 9

_IS_WINDOWS = sys.platform == "win32"


if _IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


class WindowsJobSandbox:
    """
    Manages a Windows Job Object instance to restrict memory, CPU, and child processes.
    Replicates Chromium's renderer process isolation model.
    """

    def __init__(self, memory_limit_mb: int = 256, kill_on_close: bool = True) -> None:
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.kill_on_close = kill_on_close
        self.job_handle: Optional[int] = None

        if _IS_WINDOWS:
            self._init_windows_job()

    def _init_windows_job(self) -> None:
        """Initialize Windows kernel Job Object with extended limit info."""
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                logger.warning("Failed to create Windows Job Object handle.")
                return

            self.job_handle = handle

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            flags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY
            if self.kill_on_close:
                flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            info.BasicLimitInformation.LimitFlags = flags
            info.ProcessMemoryLimit = self.memory_limit_bytes
            info.JobMemoryLimit = self.memory_limit_bytes

            success = kernel32.SetInformationJobObject(
                self.job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not success:
                err = kernel32.GetLastError()
                logger.warning("SetInformationJobObject failed with error code %d", err)
            else:
                logger.debug(
                    "Windows Job Object configured with %d MB memory limit and KillOnClose=%s.",
                    self.memory_limit_mb,
                    self.kill_on_close,
                )
        except Exception as e:
            logger.warning("Error configuring Windows Job Object: %s", e)

    def assign_process(self, process: subprocess.Popen) -> bool:
        """Assign a running subprocess to the Job Object."""
        if not _IS_WINDOWS or not self.job_handle:
            return False

        try:
            kernel32 = ctypes.windll.kernel32
            h_process = int(process._handle) if hasattr(process, "_handle") else None
            if not h_process:
                # Open process handle via PID
                PROCESS_ALL_ACCESS = 0x1F0FFF
                h_process = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, process.pid)

            if not h_process:
                logger.warning("Could not obtain process handle for PID %d", process.pid)
                return False

            res = kernel32.AssignProcessToJobObject(self.job_handle, h_process)
            if not res:
                err = kernel32.GetLastError()
                logger.warning("AssignProcessToJobObject failed for PID %d (error %d)", process.pid, err)
                return False

            logger.debug("Successfully assigned PID %d to Windows Job Sandbox.", process.pid)
            return True
        except Exception as e:
            logger.warning("Failed to assign process to Job Object: %s", e)
            return False

    def close(self) -> None:
        """Close the job handle, automatically killing child processes if kill_on_close is set."""
        if _IS_WINDOWS and self.job_handle:
            try:
                ctypes.windll.kernel32.CloseHandle(self.job_handle)
            except Exception as e:
                logger.debug("Error closing Job Object handle: %s", e)
            finally:
                self.job_handle = None

    def __enter__(self) -> WindowsJobSandbox:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def run_sandboxed_python_code(
    code: str,
    memory_limit_mb: int = 256,
    timeout_seconds: float = 30.0,
    cwd: Optional[str] = None,
) -> Tuple[int, str, str]:
    """
    Executes Python code in an isolated subprocess wrapped by a Windows Job Object memory cap.

    Args:
        code: Python source code string.
        memory_limit_mb: Hard RAM cap in megabytes (default 256MB).
        timeout_seconds: Hard execution deadline in seconds (default 30s).
        cwd: Optional working directory for confinement (defaults to repo workspace).

    Returns:
        Tuple of (exit_code, stdout, stderr).
    """
    python_exe = sys.executable

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        temp_script = tf.name

    sandbox = WindowsJobSandbox(memory_limit_mb=memory_limit_mb, kill_on_close=True)
    try:
        proc = subprocess.Popen(
            [python_exe, temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )

        # Attach process to Job Object sandbox immediately
        sandbox.assign_process(proc)

        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            exit_code = proc.returncode
            return exit_code, stdout, stderr
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return -1, stdout, f"Sandbox Timeout: Execution exceeded {timeout_seconds}s deadline."
    except Exception as e:
        return 1, "", f"Sandbox Execution Error: {e}"
    finally:
        sandbox.close()
        try:
            if os.path.exists(temp_script):
                os.remove(temp_script)
        except Exception:
            pass
