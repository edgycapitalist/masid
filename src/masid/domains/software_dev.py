"""Software Development task domain.

Tasks involve building small Python utilities or modules.
Evaluation is semi-automated: code correctness, structure quality,
test coverage, and review coherence.
"""

from __future__ import annotations

from masid.domains import TaskSpec

SOFTWARE_DEV_TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="sw_001",
        title="URL Shortener Library",
        description=(
            "Design and implement a Python library that provides URL shortening "
            "functionality. The library should:\n"
            "1. Accept a long URL and return a shortened version using a hash-based "
            "   approach (no external services).\n"
            "2. Support a reverse lookup from short URL to original URL.\n"
            "3. Handle collision detection.\n"
            "4. Include input validation (valid URL format).\n"
            "5. Be thread-safe for concurrent usage.\n\n"
            "Deliverables: architecture document, implementation code, unit tests, "
            "and code review."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A well-structured Python module with a URLShortener class, hash-based "
            "encoding, a storage backend (dict or similar), comprehensive tests, "
            "and a thoughtful architecture document."
        ),
    ),
    TaskSpec(
        task_id="sw_002",
        title="Rate Limiter Module",
        description=(
            "Design and implement a Python rate limiter module that supports:\n"
            "1. Token bucket algorithm for rate limiting.\n"
            "2. Configurable rate (requests per second) and burst size.\n"
            "3. Multiple named rate limiters (e.g., per-user, per-endpoint).\n"
            "4. A decorator for easy application to functions.\n"
            "5. Thread-safe operation.\n\n"
            "Deliverables: architecture document, implementation code, unit tests, "
            "and code review."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A token bucket implementation with configurable parameters, a registry "
            "for named limiters, a @rate_limit decorator, proper locking, and "
            "thorough tests including concurrency scenarios."
        ),
    ),
    TaskSpec(
        task_id="sw_003",
        title="CSV Data Pipeline",
        description=(
            "Design and implement a Python data pipeline that:\n"
            "1. Reads CSV files with configurable schema validation.\n"
            "2. Applies a chain of transformations (filtering, mapping, aggregation).\n"
            "3. Supports a plugin system for custom transformations.\n"
            "4. Outputs results to CSV or JSON.\n"
            "5. Provides clear error reporting for malformed data.\n\n"
            "Deliverables: architecture document, implementation code, unit tests, "
            "and code review."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A pipeline class with reader/transformer/writer stages, a plugin "
            "interface for custom transforms, schema validation, error collection, "
            "and tests covering happy path and error cases."
        ),
    ),
    TaskSpec(
        task_id="sw_004",
        title="Simple Task Queue",
        description=(
            "Design and implement an in-memory task queue in Python that:\n"
            "1. Supports adding tasks with priorities.\n"
            "2. Allows worker threads to consume tasks.\n"
            "3. Provides task status tracking (pending, running, completed, failed).\n"
            "4. Supports task retry with configurable max retries.\n"
            "5. Includes graceful shutdown.\n\n"
            "Deliverables: architecture document, implementation code, unit tests, "
            "and code review."
        ),
        difficulty="hard",
        expected_output_hint=(
            "A priority queue backed by heapq, worker thread pool, task state "
            "machine, retry logic with backoff, shutdown signaling, and tests "
            "including concurrent producer/consumer scenarios."
        ),
    ),
]


def get_software_dev_tasks() -> list[TaskSpec]:
    """Return all software development task specifications."""
    return SOFTWARE_DEV_TASKS
