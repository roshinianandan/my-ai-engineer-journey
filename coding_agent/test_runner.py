import os
import re
import time
import tempfile
import subprocess
import sys
import ollama
from coding_agent.executor import CodeExecutor
from coding_agent.code_generator import CodeGenerator, extract_code
from config import MODEL


class TestDrivenAgent:
    """
    Test-Driven Development agent:
    1. Write tests for the task first
    2. Generate function code
    3. Run tests against the code
    4. Fix code if tests fail
    5. Repeat until all tests pass

    TDD produces more reliable code than generate-then-test
    because the tests define the exact expected behavior upfront.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        verbose: bool = True
    ):
        self.max_attempts = max_attempts
        self.executor = CodeExecutor(timeout_seconds=20)
        self.generator = CodeGenerator()
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def run_tests(
        self,
        function_code: str,
        test_code: str
    ) -> dict:
        """
        Run pytest tests against generated function code.
        Combines function code and test code into one file.
        """
        combined_code = f"""
{function_code}

{test_code}

# Run tests manually if pytest not available
import sys

def run_manual_tests():
    test_functions = [
        name for name in dir()
        if name.startswith('test_')
    ]
    passed = 0
    failed = 0
    errors = []
    for test_name in test_functions:
        try:
            test_fn = globals()[test_name]
            test_fn()
            passed += 1
            print(f"PASSED: {{test_name}}")
        except AssertionError as e:
            failed += 1
            errors.append(f"FAILED: {{test_name}} - {{e}}")
            print(f"FAILED: {{test_name}} - {{e}}")
        except Exception as e:
            failed += 1
            errors.append(f"ERROR: {{test_name}} - {{e}}")
            print(f"ERROR: {{test_name}} - {{e}}")

    print(f"\\nResults: {{passed}} passed, {{failed}} failed")
    if failed > 0:
        sys.exit(1)

run_manual_tests()
"""
        result = self.executor.execute(combined_code)
        passed = result.stdout.count("PASSED:")
        failed = result.stdout.count("FAILED:") + result.stdout.count("ERROR:")

        return {
            "success": result.success and failed == 0,
            "passed": passed,
            "failed": failed,
            "output": result.stdout,
            "error": result.stderr,
            "exit_code": result.exit_code
        }

    def solve_tdd(
        self,
        task: str,
        function_name: str = "solution"
    ) -> dict:
        """
        Solve a task using Test-Driven Development.
        Generates tests first, then iterates until they pass.
        """
        self._log(f"\n{'='*60}")
        self._log(f"  TEST-DRIVEN CODING AGENT")
        self._log(f"  Task: {task}")
        self._log(f"  Function: {function_name}")
        self._log(f"{'='*60}\n")

        start_time = time.time()

        # Step 1: Generate tests
        self._log("[Step 1] Generating test suite...")
        function_sig = f"def {function_name}(...):"
        test_code = self.generator.generate_tests(task, function_sig)
        self._log(f"Generated tests:\n{'-'*40}")
        self._log(test_code)
        self._log(f"{'-'*40}\n")

        # Step 2: Generate initial function
        self._log("[Step 2] Generating initial function...")
        function_code = self.generator.generate_function(task, function_name)
        self._log(f"Generated function:\n{'-'*40}")
        self._log(function_code)
        self._log(f"{'-'*40}\n")

        history = []

        for attempt in range(1, self.max_attempts + 1):
            self._log(f"\n[Attempt {attempt}/{self.max_attempts}] Running tests...")

            test_result = self.run_tests(function_code, test_code)

            history.append({
                "attempt": attempt,
                "function_code": function_code,
                "test_result": test_result
            })

            self._log(f"  Tests: {test_result['passed']} passed, "
                      f"{test_result['failed']} failed")

            if test_result["success"]:
                elapsed = round(time.time() - start_time, 2)
                self._log(f"\n✅ All tests pass! (attempt {attempt})")
                self._log(f"   Time: {elapsed}s")

                return {
                    "task": task,
                    "success": True,
                    "function_code": function_code,
                    "test_code": test_code,
                    "tests_passed": test_result["passed"],
                    "attempts": attempt,
                    "total_time": elapsed,
                    "history": history
                }

            # Fix the function based on test failures
            self._log(f"  Fixing function based on test failures...")
            error_context = (
                f"Test output:\n{test_result['output']}\n"
                f"Error:\n{test_result['error']}"
            )
            function_code = self.generator.fix_code(
                code=function_code,
                error_message=error_context,
                task_description=task
            )

        elapsed = round(time.time() - start_time, 2)
        self._log(f"\n💀 Failed after {self.max_attempts} attempts ({elapsed}s)")

        return {
            "task": task,
            "success": False,
            "function_code": function_code,
            "test_code": test_code,
            "attempts": self.max_attempts,
            "total_time": elapsed,
            "history": history
        }


def demo_coding_agent():
    """Run the coding agent on a set of demo tasks."""
    from coding_agent.debugger import SelfDebuggingAgent

    agent = SelfDebuggingAgent(max_attempts=4, verbose=True)

    tasks = [
        "Write a Python function that checks if a number is prime and print whether 17, 20, and 97 are prime",
        "Write code to find the first 10 Fibonacci numbers and print them",
        "Write a function that counts word frequency in a string and test it with 'hello world hello python world hello'",
    ]

    for task in tasks:
        print(f"\n{'#'*60}")
        result = agent.solve(task)
        if result["success"]:
            print(f"\n✅ Solved in {result['attempts']} attempts!")
            print(f"Output:\n{result['output']}")
        else:
            print(f"\n❌ Could not solve after {result['attempts']} attempts")
        print(f"{'#'*60}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test-Driven Coding Agent")
    parser.add_argument("--task",  type=str, help="Task to solve")
    parser.add_argument("--tdd",   type=str, help="Solve using TDD")
    parser.add_argument("--demo",  action="store_true", help="Run demo tasks")
    parser.add_argument("--fn",    type=str, default="solution",
                        help="Function name for TDD mode")
    args = parser.parse_args()

    if args.demo:
        demo_coding_agent()
    elif args.tdd:
        agent = TestDrivenAgent(max_attempts=4, verbose=True)
        result = agent.solve_tdd(args.tdd, function_name=args.fn)
        if result["success"]:
            print(f"\n✅ All {result['tests_passed']} tests pass!")
            print(f"\nFinal code:\n{result['function_code']}")
        else:
            print(f"\n❌ Tests still failing after {result['attempts']} attempts")
    elif args.task:
        from coding_agent.debugger import SelfDebuggingAgent
        agent = SelfDebuggingAgent(max_attempts=4, verbose=True)
        result = agent.solve(args.task)
        if result["success"]:
            print(f"\n✅ Solved in {result['attempts']} attempts!")
            print(f"\nOutput:\n{result['output']}")
            print(f"\nFinal code:\n{result['code']}")
    else:
        parser.print_help()