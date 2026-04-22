import time
from coding_agent.executor import CodeExecutor, ExecutionResult
from coding_agent.code_generator import CodeGenerator


class SelfDebuggingAgent:
    """
    An agent that generates code, runs it, and fixes errors automatically.

    The debugging loop:
    1. Generate code for the task
    2. Execute the code
    3. If success → return the working code
    4. If error → send error back to LLM to fix
    5. Repeat until success or max attempts reached

    This is the core pattern behind AI coding tools like Devin and GitHub Copilot.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        timeout_seconds: int = 15,
        verbose: bool = True
    ):
        self.max_attempts = max_attempts
        self.executor = CodeExecutor(timeout_seconds=timeout_seconds)
        self.generator = CodeGenerator()
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def solve(self, task: str) -> dict:
        """
        Solve a coding task with automatic error fixing.

        Returns:
        - code: the final working code
        - success: whether the code ran without errors
        - attempts: number of attempts taken
        - output: the program's output
        - history: list of all attempts
        """
        self._log(f"\n{'='*60}")
        self._log(f"  SELF-DEBUGGING CODING AGENT")
        self._log(f"  Task: {task}")
        self._log(f"  Max attempts: {self.max_attempts}")
        self._log(f"{'='*60}\n")

        history = []
        start_time = time.time()
        current_code = None

        for attempt in range(1, self.max_attempts + 1):
            self._log(f"\n[Attempt {attempt}/{self.max_attempts}]")

            # Generate or fix code
            if current_code is None:
                self._log("Generating initial code...")
                current_code = self.generator.generate_script(task)
            else:
                last_result = history[-1]["result"]
                error = last_result.stderr or f"Exit code: {last_result.exit_code}"
                self._log(f"Fixing error: {error[:100]}...")
                current_code = self.generator.fix_code(
                    code=current_code,
                    error_message=error,
                    task_description=task
                )

            self._log(f"\nGenerated code:\n{'-'*40}")
            self._log(current_code)
            self._log(f"{'-'*40}")

            # Execute the code
            self._log(f"\nExecuting code...")
            result = self.executor.execute(current_code)

            history.append({
                "attempt": attempt,
                "code": current_code,
                "result": result
            })

            if result.success:
                elapsed = round(time.time() - start_time, 2)
                self._log(f"\n✅ SUCCESS on attempt {attempt}!")
                self._log(f"   Output: {result.stdout[:200]}")
                self._log(f"   Time: {result.execution_time_ms}ms")
                self._log(f"   Total time: {elapsed}s")

                return {
                    "task": task,
                    "success": True,
                    "code": current_code,
                    "output": result.stdout,
                    "attempts": attempt,
                    "total_time": elapsed,
                    "history": history
                }
            else:
                self._log(f"\n❌ Attempt {attempt} failed:")
                self._log(f"   Error: {result.stderr[:200]}")

        # All attempts failed
        elapsed = round(time.time() - start_time, 2)
        self._log(f"\n💀 Failed after {self.max_attempts} attempts")

        return {
            "task": task,
            "success": False,
            "code": current_code,
            "output": "",
            "error": history[-1]["result"].stderr if history else "Unknown error",
            "attempts": self.max_attempts,
            "total_time": elapsed,
            "history": history
        }

    def solve_with_validation(
        self,
        task: str,
        validator_fn = None
    ) -> dict:
        """
        Solve a task with a custom validation function.

        validator_fn: takes (code, stdout) and returns (is_valid, feedback)
        Useful when you need to check more than just "did it run without errors?"
        """
        self._log(f"\n[Agent] Solving with custom validation: {task[:60]}...")

        for attempt in range(1, self.max_attempts + 1):
            self._log(f"\n[Attempt {attempt}]")

            if attempt == 1:
                code = self.generator.generate_script(task)
            else:
                last = history[-1]
                feedback = last.get("feedback", last["result"].stderr)
                code = self.generator.fix_code(
                    code=last["code"],
                    error_message=feedback,
                    task_description=task
                )

            result = self.executor.execute(code)
            history = getattr(self, '_history', [])
            entry = {"attempt": attempt, "code": code, "result": result}

            if not result.success:
                entry["feedback"] = result.stderr
                history.append(entry)
                self._log(f"  ❌ Execution error: {result.stderr[:100]}")
                continue

            # Run custom validator
            if validator_fn:
                is_valid, feedback = validator_fn(code, result.stdout)
                if not is_valid:
                    entry["feedback"] = feedback
                    history.append(entry)
                    self._log(f"  ❌ Validation failed: {feedback}")
                    continue

            history.append(entry)
            self._log(f"  ✅ Success on attempt {attempt}")
            self._history = []
            return {
                "success": True,
                "code": code,
                "output": result.stdout,
                "attempts": attempt
            }

        self._history = []
        return {
            "success": False,
            "code": code,
            "attempts": self.max_attempts
        }