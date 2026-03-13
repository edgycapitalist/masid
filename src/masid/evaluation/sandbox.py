"""Sandboxed code execution for the software development domain.

Extracts Python code from agent outputs (markdown fences), writes to
temp files, and executes in isolated subprocesses with timeouts.
Returns objective metrics: syntax validity, execution success, test
pass/fail counts.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Safety limits
_EXECUTION_TIMEOUT = 30  # seconds
_MAX_OUTPUT_CHARS = 10_000


@dataclass(frozen=True)
class CodeExtractionResult:
    """Result of extracting code blocks from agent output."""

    source_code: str  # Combined implementation code
    test_code: str  # Combined test code
    num_code_blocks: int
    num_test_blocks: int


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing code in the sandbox."""

    # Syntax
    syntax_valid: bool
    syntax_error: str = ""

    # Execution
    code_runs: bool = False
    code_output: str = ""
    code_error: str = ""

    # Tests
    tests_run: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errors: int = 0
    tests_total: int = 0
    test_output: str = ""

    # Summary score (0-1)
    execution_score: float = 0.0


def extract_code_blocks(text: str) -> CodeExtractionResult:
    """Extract Python code blocks from markdown-formatted agent output.

    Looks for ```python ... ``` fences (and plain ``` ... ``` fences).
    Classifies blocks as test code if they contain test markers.

    Strategy: Instead of concatenating all source blocks (which often
    breaks due to duplicate imports/classes), we select the LARGEST
    source block as the primary implementation. If multiple test blocks
    exist, we concatenate them (tests are additive).
    """
    # Match both ```python and plain ``` fences
    pattern = r"```(?:python|py)?\s*\n?(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)

    source_blocks = []
    test_blocks = []

    for block in blocks:
        block = block.strip()
        if not block or len(block) < 10:
            continue

        # Skip blocks that are clearly not Python (shell commands, etc.)
        if block.startswith("$") or block.startswith("pip ") or block.startswith("bash"):
            continue

        # Classify as test code or source code
        is_test = any(marker in block for marker in [
            "def test_", "import unittest", "import pytest",
            "from unittest", "class Test",
        ])
        if is_test:
            test_blocks.append(block)
        else:
            source_blocks.append(block)

    # Select the largest source block as the primary implementation.
    # Models often output code in multiple blocks (e.g., "here's the class"
    # then "here's the usage example"). The largest block is typically
    # the complete implementation.
    if source_blocks:
        best_source = max(source_blocks, key=len)
    else:
        best_source = ""

    # For tests, concatenate all test blocks (they're additive).
    # But deduplicate imports.
    if test_blocks:
        combined_tests = _merge_test_blocks(test_blocks)
    else:
        combined_tests = ""

    return CodeExtractionResult(
        source_code=best_source,
        test_code=combined_tests,
        num_code_blocks=len(source_blocks),
        num_test_blocks=len(test_blocks),
    )


def _merge_test_blocks(blocks: list[str]) -> str:
    """Merge multiple test code blocks, deduplicating imports."""
    if len(blocks) == 1:
        return blocks[0]

    imports = set()
    body_lines = []

    for block in blocks:
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                imports.add(stripped)
            else:
                body_lines.append(line)

    merged = "\n".join(sorted(imports)) + "\n\n" + "\n".join(body_lines)
    return merged.strip()


def check_syntax(code: str) -> tuple[bool, str]:
    """Check if Python code has valid syntax without executing it.

    Returns
    -------
    (is_valid, error_message)
    """
    if not code.strip():
        return False, "Empty code"
    try:
        compile(code, "<agent_code>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def run_code(code: str, timeout: int = _EXECUTION_TIMEOUT) -> tuple[bool, str, str]:
    """Execute Python code in an isolated subprocess.

    Returns
    -------
    (success, stdout, stderr)
    """
    if not code.strip():
        return False, "", "Empty code"

    with tempfile.TemporaryDirectory() as tmpdir:
        code_path = Path(tmpdir) / "agent_code.py"
        code_path.write_text(code)

        try:
            result = subprocess.run(
                ["python3", str(code_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout = result.stdout[:_MAX_OUTPUT_CHARS]
            stderr = result.stderr[:_MAX_OUTPUT_CHARS]
            return result.returncode == 0, stdout, stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Execution timed out after {timeout}s"
        except Exception as e:
            return False, "", f"Execution error: {e}"


def run_tests(
    source_code: str,
    test_code: str,
    timeout: int = _EXECUTION_TIMEOUT,
) -> tuple[bool, int, int, int, str]:
    """Execute tests against source code in an isolated subprocess.

    Writes source code to a module file, writes test code to a test
    file that imports the module, and runs pytest.

    Returns
    -------
    (tests_ran, passed, failed, errors, output)
    """
    if not source_code.strip() or not test_code.strip():
        return False, 0, 0, 0, "Missing source or test code"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source code as importable module
        module_path = Path(tmpdir) / "solution.py"
        module_path.write_text(source_code)

        # Write test code — prepend import of the solution module
        # and add sys.path manipulation so imports work
        test_preamble = (
            "import sys\n"
            f"sys.path.insert(0, {str(tmpdir)!r})\n"
        )
        test_path = Path(tmpdir) / "test_solution.py"
        test_path.write_text(test_preamble + test_code)

        try:
            # Try running with pytest first (more reliable parsing)
            result = subprocess.run(
                ["python3", "-m", "pytest", str(test_path), "-v", "--tb=short", "--no-header"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = result.stdout[:_MAX_OUTPUT_CHARS]
            stderr = result.stderr[:_MAX_OUTPUT_CHARS]

            # Parse pytest output for pass/fail counts
            passed, failed, errors = _parse_pytest_output(output + "\n" + stderr)
            tests_ran = (passed + failed + errors) > 0

            return tests_ran, passed, failed, errors, output

        except subprocess.TimeoutExpired:
            return False, 0, 0, 0, f"Tests timed out after {timeout}s"
        except Exception as e:
            return False, 0, 0, 0, f"Test execution error: {e}"


def _parse_pytest_output(output: str) -> tuple[int, int, int]:
    """Parse pytest output to extract pass/fail/error counts.

    Returns (passed, failed, errors).
    """
    passed = failed = errors = 0

    # Look for the summary line like "3 passed, 1 failed, 1 error"
    summary = re.search(
        r"(\d+)\s+passed",
        output,
    )
    if summary:
        passed = int(summary.group(1))

    fail_match = re.search(r"(\d+)\s+failed", output)
    if fail_match:
        failed = int(fail_match.group(1))

    err_match = re.search(r"(\d+)\s+error", output)
    if err_match:
        errors = int(err_match.group(1))

    return passed, failed, errors


def evaluate_agent_code(
    coder_output: str,
    tester_output: str,
) -> ExecutionResult:
    """Full evaluation pipeline for the software dev domain.

    1. Extract code from Coder's output
    2. Extract tests from Tester's output
    3. Check syntax of both
    4. Run the code
    5. Run the tests against the code
    6. Compute an objective execution score

    Parameters
    ----------
    coder_output : str
        Raw text output from the Coder agent.
    tester_output : str
        Raw text output from the Tester agent.

    Returns
    -------
    ExecutionResult
    """
    # Extract code blocks
    code_extraction = extract_code_blocks(coder_output)
    test_extraction = extract_code_blocks(tester_output)

    source_code = code_extraction.source_code
    test_code = test_extraction.test_code

    # If tester didn't put tests in fenced blocks, try extracting from
    # their full output (some models write tests inline)
    if not test_code:
        test_extraction_full = extract_code_blocks(tester_output)
        # Use any code blocks from tester as tests
        test_code = test_extraction_full.source_code

    logger.info(
        "  Code extraction: %d source blocks, %d test blocks",
        code_extraction.num_code_blocks,
        test_extraction.num_test_blocks or (1 if test_code else 0),
    )

    # Check syntax
    syntax_valid, syntax_error = check_syntax(source_code)
    if not syntax_valid:
        logger.info("  Syntax check failed: %s", syntax_error)
        return ExecutionResult(
            syntax_valid=False,
            syntax_error=syntax_error,
            execution_score=0.1,  # Small credit for producing something
        )

    # Run the code (check for import/runtime errors)
    code_runs, code_output, code_error = run_code(source_code)
    logger.info("  Code execution: %s", "OK" if code_runs else f"FAILED ({code_error[:80]})")

    # Run tests
    tests_run = False
    tests_passed = tests_failed = tests_errors = 0
    test_output = ""

    if test_code.strip():
        tests_run, tests_passed, tests_failed, tests_errors, test_output = run_tests(
            source_code, test_code
        )
        logger.info(
            "  Tests: ran=%s passed=%d failed=%d errors=%d",
            tests_run, tests_passed, tests_failed, tests_errors,
        )
    else:
        logger.info("  No test code extracted from Tester output")

    # Compute objective score
    score = _compute_execution_score(
        syntax_valid=syntax_valid,
        code_runs=code_runs,
        tests_run=tests_run,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errors=tests_errors,
    )

    return ExecutionResult(
        syntax_valid=syntax_valid,
        syntax_error=syntax_error,
        code_runs=code_runs,
        code_output=code_output[:500],
        code_error=code_error[:500],
        tests_run=tests_run,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_errors=tests_errors,
        tests_total=tests_passed + tests_failed + tests_errors,
        test_output=test_output[:500],
        execution_score=score,
    )


def _compute_execution_score(
    syntax_valid: bool,
    code_runs: bool,
    tests_run: bool,
    tests_passed: int,
    tests_failed: int,
    tests_errors: int,
) -> float:
    """Compute a 0-1 objective score from execution results.

    Scoring breakdown:
    - 0.20: Valid syntax
    - 0.30: Code executes without errors
    - 0.50: Test results (proportional to pass rate)
    """
    score = 0.0

    # Syntax validity (20%)
    if syntax_valid:
        score += 0.20

    # Code runs (30%)
    if code_runs:
        score += 0.30

    # Test pass rate (50%)
    if tests_run:
        total = tests_passed + tests_failed + tests_errors
        if total > 0:
            pass_rate = tests_passed / total
            score += 0.50 * pass_rate

    return round(score, 4)