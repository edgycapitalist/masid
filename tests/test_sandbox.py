"""Tests for masid.evaluation.sandbox."""


from masid.evaluation.sandbox import (
    _compute_execution_score,
    _parse_pytest_output,
    check_syntax,
    evaluate_agent_code,
    extract_code_blocks,
    run_code,
    run_tests,
)


class TestExtractCodeBlocks:
    def test_single_python_block(self):
        text = "Here is the code:\n```python\ndef hello():\n    return 'world'\n```\nDone."
        result = extract_code_blocks(text)
        assert result.num_code_blocks == 1
        assert "def hello" in result.source_code

    def test_multiple_blocks(self):
        text = (
            "Implementation:\n```python\nclass Foo:\n    pass\n```\n"
            "Tests:\n```python\ndef test_foo():\n    assert True\n```\n"
        )
        result = extract_code_blocks(text)
        assert result.num_code_blocks == 1
        assert result.num_test_blocks == 1
        assert "class Foo" in result.source_code
        assert "def test_foo" in result.test_code

    def test_no_code_blocks(self):
        text = "Here is my analysis of the code. It looks good."
        result = extract_code_blocks(text)
        assert result.num_code_blocks == 0
        assert result.source_code == ""

    def test_unfenced_code_ignored(self):
        text = "def hello():\n    return 'world'\n"
        result = extract_code_blocks(text)
        assert result.num_code_blocks == 0

    def test_largest_block_selected(self):
        """When multiple source blocks exist, the largest is selected."""
        text = (
            "Helper:\n```python\nimport os\nprint('small')\n```\n"
            "Full implementation:\n```python\n"
            "import hashlib\n\nclass URLShortener:\n"
            "    def __init__(self):\n        self.store = {}\n"
            "    def shorten(self, url):\n        return hashlib.md5(url.encode()).hexdigest()[:8]\n"
            "```\n"
        )
        result = extract_code_blocks(text)
        assert "class URLShortener" in result.source_code
        assert "print('small')" not in result.source_code

    def test_test_classification(self):
        text = "```python\nimport pytest\ndef test_something():\n    assert 1 == 1\n```"
        result = extract_code_blocks(text)
        assert result.num_test_blocks == 1
        assert result.num_code_blocks == 0


class TestCheckSyntax:
    def test_valid_syntax(self):
        valid, error = check_syntax("def foo():\n    return 42\n")
        assert valid is True
        assert error == ""

    def test_invalid_syntax(self):
        valid, error = check_syntax("def foo(\n    return 42\n")
        assert valid is False
        assert "SyntaxError" in error or error != ""

    def test_empty_code(self):
        valid, error = check_syntax("")
        assert valid is False


class TestRunCode:
    def test_simple_execution(self):
        success, stdout, stderr = run_code("print('hello')")
        assert success is True
        assert "hello" in stdout

    def test_runtime_error(self):
        success, stdout, stderr = run_code("raise ValueError('oops')")
        assert success is False
        assert "ValueError" in stderr

    def test_import_error(self):
        success, stdout, stderr = run_code("import nonexistent_module_xyz")
        assert success is False

    def test_timeout(self):
        success, stdout, stderr = run_code("import time; time.sleep(100)", timeout=2)
        assert success is False
        assert "timed out" in stderr.lower()


class TestRunTests:
    def test_passing_tests(self):
        source = "def add(a, b):\n    return a + b\n"
        tests = (
            "from solution import add\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
            "def test_add_zero():\n"
            "    assert add(0, 0) == 0\n"
        )
        ran, passed, failed, errors, output = run_tests(source, tests)
        assert ran is True
        assert passed == 2
        assert failed == 0

    def test_failing_tests(self):
        source = "def add(a, b):\n    return a - b  # bug!\n"
        tests = (
            "from solution import add\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
        ran, passed, failed, errors, output = run_tests(source, tests)
        assert ran is True
        assert failed >= 1

    def test_empty_source(self):
        ran, passed, failed, errors, output = run_tests("", "def test_x(): pass")
        assert ran is False


class TestParsePytestOutput:
    def test_all_passed(self):
        output = "===== 5 passed in 0.03s ====="
        p, f, e = _parse_pytest_output(output)
        assert p == 5 and f == 0 and e == 0

    def test_mixed_results(self):
        output = "===== 3 passed, 2 failed, 1 error in 0.1s ====="
        p, f, e = _parse_pytest_output(output)
        assert p == 3 and f == 2 and e == 1

    def test_no_results(self):
        p, f, e = _parse_pytest_output("no tests ran")
        assert p == 0 and f == 0 and e == 0


class TestComputeExecutionScore:
    def test_perfect_score(self):
        score = _compute_execution_score(
            syntax_valid=True, code_runs=True,
            tests_run=True, tests_passed=5, tests_failed=0, tests_errors=0,
        )
        assert score == 1.0

    def test_syntax_only(self):
        score = _compute_execution_score(
            syntax_valid=True, code_runs=False,
            tests_run=False, tests_passed=0, tests_failed=0, tests_errors=0,
        )
        assert score == 0.2

    def test_no_tests(self):
        score = _compute_execution_score(
            syntax_valid=True, code_runs=True,
            tests_run=False, tests_passed=0, tests_failed=0, tests_errors=0,
        )
        assert score == 0.5  # 0.2 syntax + 0.3 runs

    def test_partial_tests(self):
        score = _compute_execution_score(
            syntax_valid=True, code_runs=True,
            tests_run=True, tests_passed=3, tests_failed=1, tests_errors=0,
        )
        # 0.2 + 0.3 + 0.5*(3/4) = 0.875
        assert abs(score - 0.875) < 0.01

    def test_nothing_works(self):
        score = _compute_execution_score(
            syntax_valid=False, code_runs=False,
            tests_run=False, tests_passed=0, tests_failed=0, tests_errors=0,
        )
        assert score == 0.0


class TestEvaluateAgentCode:
    def test_good_code_and_tests(self):
        coder_output = (
            "Here is the implementation:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
        )
        tester_output = (
            "Here are the tests:\n"
            "```python\n"
            "from solution import add\n"
            "def test_add_positive():\n"
            "    assert add(2, 3) == 5\n"
            "def test_add_negative():\n"
            "    assert add(-1, -1) == -2\n"
            "```\n"
        )
        result = evaluate_agent_code(coder_output, tester_output)
        assert result.syntax_valid is True
        assert result.code_runs is True
        assert result.tests_passed >= 1
        assert result.execution_score > 0.5

    def test_bad_syntax(self):
        coder_output = "```python\ndef broken(\n    return\n```"
        tester_output = "```python\ndef test_x(): pass\n```"
        result = evaluate_agent_code(coder_output, tester_output)
        assert result.syntax_valid is False
        assert result.execution_score <= 0.1

    def test_no_code_blocks(self):
        result = evaluate_agent_code(
            "The code should do X and Y",
            "The tests should verify A and B",
        )
        assert result.syntax_valid is False
