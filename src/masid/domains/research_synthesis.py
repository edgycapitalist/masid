"""Research Synthesis task domain.

Tasks involve synthesizing findings from multiple source documents.
Evaluation uses LLM-as-judge for coherence, completeness, and accuracy.
"""

from __future__ import annotations

from masid.domains import TaskSpec

RESEARCH_SYNTHESIS_TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="rs_001",
        title="AI Regulation Landscape",
        description=(
            "Synthesize a report on the current state of AI regulation globally.\n\n"
            "Source Set A (for Researcher A):\n"
            "- The EU AI Act categorizes AI systems by risk level and imposes "
            "  requirements accordingly. High-risk systems face strict compliance.\n"
            "- China has implemented regulations on generative AI requiring "
            "  content labeling and algorithmic transparency.\n\n"
            "Source Set B (for Researcher B):\n"
            "- The US has taken an executive-order approach rather than "
            "  comprehensive legislation, focusing on safety testing.\n"
            "- The UK favors a sector-specific, principles-based framework "
            "  rather than horizontal regulation.\n\n"
            "Deliverables: individual research findings, a coherent synthesis "
            "report, and a fact-check assessment."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A well-structured synthesis comparing regulatory approaches across "
            "four jurisdictions, identifying common themes and divergences, with "
            "all claims traceable to source documents."
        ),
    ),
    TaskSpec(
        task_id="rs_002",
        title="Remote Work Productivity",
        description=(
            "Synthesize research findings on remote work and productivity.\n\n"
            "Source Set A (for Researcher A):\n"
            "- A Stanford study found that remote workers showed a 13%% performance "
            "  increase, partly from working more minutes per shift.\n"
            "- Microsoft's Work Trend Index found that collaboration networks became "
            "  more siloed during remote work.\n\n"
            "Source Set B (for Researcher B):\n"
            "- A Harvard Business School study found remote workers were 4.4%% less "
            "  productive than office workers, but hybrid workers showed no difference.\n"
            "- Gallup data shows engagement is highest for employees spending 2-3 "
            "  days in the office per week.\n\n"
            "Deliverables: individual research findings, a coherent synthesis "
            "addressing the apparent contradictions, and a fact-check assessment."
        ),
        difficulty="medium",
        expected_output_hint=(
            "A nuanced synthesis that reconciles conflicting findings by identifying "
            "mediating variables (task type, measurement method, time period), "
            "with clear attribution to each source."
        ),
    ),
    TaskSpec(
        task_id="rs_003",
        title="LLM Scaling Laws",
        description=(
            "Synthesize findings on scaling laws for large language models.\n\n"
            "Source Set A (for Researcher A):\n"
            "- Kaplan et al. (2020) established power-law relationships between "
            "  model size, dataset size, compute, and loss.\n"
            "- The Chinchilla paper (Hoffmann et al., 2022) argued that most "
            "  models were undertrained relative to their parameter count.\n\n"
            "Source Set B (for Researcher B):\n"
            "- Recent work suggests scaling laws may plateau for certain "
            "  capabilities, with emergent abilities appearing unpredictably.\n"
            "- Inference-time compute scaling (chain-of-thought, search) offers "
            "  an alternative dimension for improving performance.\n\n"
            "Deliverables: individual research findings, a coherent synthesis, "
            "and a fact-check assessment."
        ),
        difficulty="hard",
        expected_output_hint=(
            "A technically precise synthesis tracing the evolution of scaling law "
            "understanding, noting where consensus exists and where debate remains, "
            "with accurate representation of each source's claims."
        ),
    ),
]


def get_research_synthesis_tasks() -> list[TaskSpec]:
    """Return all research synthesis task specifications."""
    return RESEARCH_SYNTHESIS_TASKS
