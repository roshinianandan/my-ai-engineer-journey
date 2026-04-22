import sys
import os
import time
import tempfile
import subprocess
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of running a piece of code."""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    code: str


class CodeExecutor:
    """
    Safely executes Python code in a subprocess sandbox.

    Why subprocess instead of exec()?
    - Isolated process — crashes don't affect the main program
    - Timeout support — kills runaway code
    - Captures all output — stdout and stderr separately
    - Cannot access parent process memory
    """

    def __init__(
        self,
        timeout_seconds: int = 15,
        max_output_chars: int = 5000
    ):
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def execute(self, code: str) -> ExecutionResult:
        """
        Execute Python code in a sandboxed subprocess.
        Returns ExecutionResult with output, errors, and timing.
        """
        # Write code to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=os.getcwd()
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            stdout = result.stdout[:self.max_output_chars]
            stderr = result.stderr[:self.max_output_chars]

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=stdout,
                stderr=stderr,
                exit_code=result.returncode,
                execution_time_ms=elapsed_ms,
                code=code
            )

        except subprocess.TimeoutExpired:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"TimeoutError: Code exceeded {self.timeout_seconds}s limit",
                exit_code=-1,
                execution_time_ms=elapsed_ms,
                code=code
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1,
                execution_time_ms=0,
                code=code
            )

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    def execute_with_input(self, code: str, stdin: str = "") -> ExecutionResult:
        """Execute code with stdin input."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = f.name

        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:self.max_output_chars],
                stderr=result.stderr[:self.max_output_chars],
                exit_code=result.returncode,
                execution_time_ms=elapsed_ms,
                code=code
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"TimeoutError: exceeded {self.timeout_seconds}s",
                exit_code=-1,
                execution_time_ms=0,
                code=code
            )
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass