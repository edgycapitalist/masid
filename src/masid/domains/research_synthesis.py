"""Research Synthesis task domain.

Tasks involve synthesizing findings from multiple source documents.
Each researcher receives ONLY their assigned source set — they do
not see the other researcher's sources. The Synthesizer and
Fact_Checker see the full task description.

This information separation is critical for the experiment:
it means the Synthesizer must actually combine distinct information
from two sources, not just reorganize information everyone already has.
"""

from __future__ import annotations

from masid.domains import TaskSpec

# ---------------------------------------------------------------------------
# Helper to build role_sources for research tasks
# ---------------------------------------------------------------------------

def _build_role_sources(
    objective: str,
    source_a: str,
    source_b: str,
    full_description: str,
) -> dict[str, str]:
    """Build per-role task descriptions with information separation."""
    return {
        "Researcher_A": (
            f"{objective}\n\n"
            f"Your assigned sources:\n{source_a}\n\n"
            f"Produce structured findings based ONLY on your assigned sources."
        ),
        "Researcher_B": (
            f"{objective}\n\n"
            f"Your assigned sources:\n{source_b}\n\n"
            f"Produce structured findings based ONLY on your assigned sources."
        ),
        # Synthesizer and Fact_Checker see the full description
        "Synthesizer": full_description,
        "Fact_Checker": full_description,
    }


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

_RS001_OBJECTIVE = "Synthesize a report on the current state of AI regulation globally."
_RS001_SOURCE_A = (
    "- The EU AI Act categorizes AI systems by risk level and imposes "
    "requirements accordingly. High-risk systems face strict compliance.\n"
    "- China has implemented regulations on generative AI requiring "
    "content labeling and algorithmic transparency."
)
_RS001_SOURCE_B = (
    "- The US has taken an executive-order approach rather than "
    "comprehensive legislation, focusing on safety testing.\n"
    "- The UK favors a sector-specific, principles-based framework "
    "rather than horizontal regulation."
)
_RS001_FULL = (
    f"{_RS001_OBJECTIVE}\n\n"
    f"Source Set A:\n{_RS001_SOURCE_A}\n\n"
    f"Source Set B:\n{_RS001_SOURCE_B}\n\n"
    "Deliverables: individual research findings, a coherent synthesis "
    "report, and a fact-check assessment."
)

_RS002_OBJECTIVE = "Synthesize research findings on remote work and productivity."
_RS002_SOURCE_A = (
    "- A Stanford study found that remote workers showed a 13% performance "
    "increase, partly from working more minutes per shift.\n"
    "- Microsoft's Work Trend Index found that collaboration networks became "
    "more siloed during remote work."
)
_RS002_SOURCE_B = (
    "- A Harvard Business School study found remote workers were 4.4% less "
    "productive than office workers, but hybrid workers showed no difference.\n"
    "- Gallup data shows engagement is highest for employees spending 2-3 "
    "days in the office per week."
)
_RS002_FULL = (
    f"{_RS002_OBJECTIVE}\n\n"
    f"Source Set A:\n{_RS002_SOURCE_A}\n\n"
    f"Source Set B:\n{_RS002_SOURCE_B}\n\n"
    "Deliverables: individual research findings, a coherent synthesis "
    "addressing the apparent contradictions, and a fact-check assessment."
)

_RS003_OBJECTIVE = "Synthesize findings on scaling laws for large language models."
_RS003_SOURCE_A = (
    "- Kaplan et al. (2020) established power-law relationships between "
    "model size, dataset size, compute, and loss.\n"
    "- The Chinchilla paper (Hoffmann et al., 2022) argued that most "
    "models were undertrained relative to their parameter count."
)
_RS003_SOURCE_B = (
    "- Recent work suggests scaling laws may plateau for certain "
    "capabilities, with emergent abilities appearing unpredictably.\n"
    "- Inference-time compute scaling (chain-of-thought, search) offers "
    "an alternative dimension for improving performance."
)
_RS003_FULL = (
    f"{_RS003_OBJECTIVE}\n\n"
    f"Source Set A:\n{_RS003_SOURCE_A}\n\n"
    f"Source Set B:\n{_RS003_SOURCE_B}\n\n"
    "Deliverables: individual research findings, a coherent synthesis, "
    "and a fact-check assessment."
)


RESEARCH_SYNTHESIS_TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="rs_001",
        title="AI Regulation Landscape",
        description=_RS001_FULL,
        difficulty="medium",
        expected_output_hint=(
            "A well-structured synthesis comparing regulatory approaches across "
            "four jurisdictions, identifying common themes and divergences, with "
            "all claims traceable to source documents."
        ),
        role_sources=_build_role_sources(
            _RS001_OBJECTIVE, _RS001_SOURCE_A, _RS001_SOURCE_B, _RS001_FULL,
        ),
    ),
    TaskSpec(
        task_id="rs_002",
        title="Remote Work Productivity",
        description=_RS002_FULL,
        difficulty="medium",
        expected_output_hint=(
            "A nuanced synthesis that reconciles conflicting findings by identifying "
            "mediating variables (task type, measurement method, time period), "
            "with clear attribution to each source."
        ),
        role_sources=_build_role_sources(
            _RS002_OBJECTIVE, _RS002_SOURCE_A, _RS002_SOURCE_B, _RS002_FULL,
        ),
    ),
    TaskSpec(
        task_id="rs_003",
        title="LLM Scaling Laws",
        description=_RS003_FULL,
        difficulty="hard",
        expected_output_hint=(
            "A technically precise synthesis tracing the evolution of scaling law "
            "understanding, noting where consensus exists and where debate remains, "
            "with accurate representation of each source's claims."
        ),
        role_sources=_build_role_sources(
            _RS003_OBJECTIVE, _RS003_SOURCE_A, _RS003_SOURCE_B, _RS003_FULL,
        ),
    ),
]


def get_research_synthesis_tasks() -> list[TaskSpec]:
    """Return all research synthesis task specifications."""
    return RESEARCH_SYNTHESIS_TASKS
