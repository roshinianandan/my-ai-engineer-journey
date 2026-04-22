import re
import ollama
from config import MODEL


CODE_GENERATION_PROMPT = """You are an expert Python programmer.
Write clean, working Python code for the task described below.

Task: {task}

Requirements:
- Write only Python code
- Include necessary imports
- Add a brief comment explaining the approach
- Make the code self-contained and runnable
- Print the result or output at the end
- Do not use external libraries unless specified

Return ONLY the Python code inside a ```python code block.
No explanation before or after the code block."""


TEST_GENERATION_PROMPT = """You are an expert Python programmer.
Write pytest tests for the following task.

Task: {task}
Function to test: {function_signature}

Requirements:
- Write 3-5 test cases covering normal cases and edge cases
- Use pytest style (def test_...)
- Import the function from the module
- Make tests specific and meaningful

Return ONLY the test code inside a ```python code block."""


FUNCTION_GENERATION_PROMPT = """You are an expert Python programmer.
Write a Python function for the task described below.

Task: {task}
Function name: {function_name}

Requirements:
- Write ONLY the function (and imports if needed)
- Include a docstring
- Handle edge cases
- Do not include test code or example usage

Return ONLY the Python code inside a ```python code block."""


def extract_code(response: str) -> str:
    """
    Extract Python code from LLM response.
    Handles ```python blocks and raw code.
    """
    # Try to find ```python block
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try plain ``` block
    pattern = r"```\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Return full response if no blocks found
    return response.strip()


class CodeGenerator:
    """
    Generates Python code using an LLM.

    Capabilities:
    - Generate standalone scripts for any task
    - Generate individual functions
    - Generate test suites
    - Improve/fix existing code
    """

    def __init__(self, model: str = MODEL):
        self.model = model

    def _call_llm(self, prompt: str, temperature: float = 0.3) -> str:
        """Call the LLM and return response."""
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": temperature}
        )
        return response["message"]["content"]

    def generate_script(self, task: str) -> str:
        """Generate a complete Python script for a task."""
        print(f"[Generator] Generating script for: {task[:60]}...")
        prompt = CODE_GENERATION_PROMPT.format(task=task)
        response = self._call_llm(prompt)
        code = extract_code(response)
        print(f"[Generator] Generated {len(code.splitlines())} lines of code")
        return code

    def generate_function(
        self,
        task: str,
        function_name: str = "solution"
    ) -> str:
        """Generate a single Python function."""
        print(f"[Generator] Generating function '{function_name}'...")
        prompt = FUNCTION_GENERATION_PROMPT.format(
            task=task,
            function_name=function_name
        )
        response = self._call_llm(prompt)
        return extract_code(response)

    def generate_tests(
        self,
        task: str,
        function_signature: str
    ) -> str:
        """Generate pytest test cases for a function."""
        print(f"[Generator] Generating tests for: {function_signature}...")
        prompt = TEST_GENERATION_PROMPT.format(
            task=task,
            function_signature=function_signature
        )
        response = self._call_llm(prompt, temperature=0.2)
        return extract_code(response)

    def improve_code(
        self,
        code: str,
        improvement_request: str
    ) -> str:
        """Improve existing code based on a specific request."""
        prompt = f"""Improve the following Python code based on this request:

Improvement needed: {improvement_request}

Current code:
```python
{code}
```

Return ONLY the improved Python code in a ```python block."""

        response = self._call_llm(prompt, temperature=0.3)
        return extract_code(response)

    def fix_code(
        self,
        code: str,
        error_message: str,
        task_description: str = ""
    ) -> str:
        """Fix code that produced an error."""
        prompt = f"""Fix the following Python code that produced an error.

Original task: {task_description}

Error message:
{error_message}

Failing code:
```python
{code}
```

Analyze the error and return the FIXED Python code in a ```python block.
Do not change what the code is supposed to do — only fix the error."""

        response = self._call_llm(prompt, temperature=0.2)
        return extract_code(response)