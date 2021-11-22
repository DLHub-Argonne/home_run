"""Utility operations related to building and executing home_run shims"""
import traceback
from contextlib import contextmanager
from typing import Dict, Any
from time import monotonic
from io import StringIO
import sys


@contextmanager
def compute_wrapper(capture_output: bool = True) -> Dict[str, Any]:
    """Wraps compute for timing, output capturing, and raise protection

    Args:
        capture_output: Whether to store the stdout and stderr rather than print to screen
    Yields:
        Metadata about the execution:
            - success: Whether the code inside ran without raising an exception
            - stdout/stderr: Captured standard output and error, if requested
            - timing: Execution time for the segment in seconds
            - exc: Captured exception object
            - error_message: Exception traceback
    """

    # Code graciously adapted from QCEngine:
    #  https://github.com/MolSSI/QCEngine/blob/4a92ee6a0dcea42316e00f9f9a7c07d95cbda111/qcengine/util.py#L97

    metadata = {"stdout": None, "stderr": None, "success": True,
                "exc": None, "error_message": None, "wall_time": None}

    # Start timer
    comp_time = monotonic()

    # Capture stdout/err
    new_stdout = StringIO()
    new_stderr = StringIO()
    if capture_output:
        old_stdout, sys.stdout = sys.stdout, new_stdout
        old_stderr, sys.stderr = sys.stderr, new_stderr
    try:
        yield metadata

    # Unknown QCEngine exception likely in the Python layer, show traceback
    except Exception as exc:
        metadata["exc"] = exc
        metadata["error_message"] = traceback.format_exc()
        metadata["success"] = False

    # Place data
    metadata["wall_time"] = monotonic() - comp_time
    if capture_output:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Pull over values
        metadata["stdout"] = new_stdout.getvalue() or None
        metadata["stderr"] = new_stderr.getvalue() or None
