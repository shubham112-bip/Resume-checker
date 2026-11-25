"""LangGraph state graph definition and workflow topology."""

from __future__ import annotations

from typing import Literal

from app.agents.nodes import (
    ats_evaluation_node,
    error_recovery_node,
    explanation_node,
    feature_engineering_node,
    ml_evaluation_node,
    recommendation_node,
    reflection_node,
    resume_parsing_node,
    review_agent_node,
    semantic_matching_node,
)
from app.agents.state import CandidateAgentState
from langgraph.graph import END, START, StateGraph


def _route_after_parse(state: CandidateAgentState) -> Literal["error_recovery", "ats_evaluation"]:
    if state.parsing_failed:
        return "error_recovery"
    return "ats_evaluation"


def _route_after_error_recovery(state: CandidateAgentState) -> Literal["ats_evaluation", "__end__"]:
    if state.parsing_failed:
        return "__end__"
    return "ats_evaluation"


def _route_after_semantic(state: CandidateAgentState) -> Literal["reflection", "feature_engineering"]:
    if state.needs_reflection:
        return "reflection"
    return "feature_engineering"


def _route_after_recommendation(state: CandidateAgentState) -> Literal["review_agent", "explanation"]:
    if len(state.review_flags) > 0 or (state.stuffing and state.stuffing.suspected):
        return "review_agent"
    return "explanation"


def build_candidate_graph() -> StateGraph:
    builder = StateGraph(CandidateAgentState)

    # 1. Add agent nodes
    builder.add_node("resume_parser", resume_parsing_node)
    builder.add_node("error_recovery", error_recovery_node)
    builder.add_node("ats_evaluation", ats_evaluation_node)
    builder.add_node("semantic_matching", semantic_matching_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("feature_engineering", feature_engineering_node)
    builder.add_node("ml_evaluation", ml_evaluation_node)
    builder.add_node("recommendation", recommendation_node)
    builder.add_node("review_agent", review_agent_node)
    builder.add_node("explanation", explanation_node)

    # 2. Add edges & conditional transitions
    builder.add_edge(START, "resume_parser")

    builder.add_conditional_edges(
        "resume_parser",
        _route_after_parse,
        {"error_recovery": "error_recovery", "ats_evaluation": "ats_evaluation"},
    )

    builder.add_conditional_edges(
        "error_recovery",
        _route_after_error_recovery,
        {"ats_evaluation": "ats_evaluation", "__end__": END},
    )

    builder.add_edge("ats_evaluation", "semantic_matching")

    builder.add_conditional_edges(
        "semantic_matching",
        _route_after_semantic,
        {"reflection": "reflection", "feature_engineering": "feature_engineering"},
    )

    builder.add_edge("reflection", "feature_engineering")
    builder.add_edge("feature_engineering", "ml_evaluation")
    builder.add_edge("ml_evaluation", "recommendation")

    builder.add_conditional_edges(
        "recommendation",
        _route_after_recommendation,
        {"review_agent": "review_agent", "explanation": "explanation"},
    )

    builder.add_edge("review_agent", "explanation")
    builder.add_edge("explanation", END)

    return builder


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        builder = build_candidate_graph()
        _compiled_graph = builder.compile()
    return _compiled_graph


def get_graph_mermaid() -> str:
    """Export the agent state graph as a Mermaid diagram for UI rendering and viva."""
    return """graph TD
    START([START: Upload Resume]) --> RP[Resume Parsing Agent]
    
    RP -->|Parse Error| ER[Error Recovery Agent]
    RP -->|Success| ATS[ATS Evaluation Agent]
    
    ER -->|Recovery Success| ATS
    ER -->|Corrupt / Scanned| FAIL([END: Parsing Failed])
    
    ATS --> SM[Semantic Matching Agent]
    
    SM -->|Uncertain / Implied Skills| REF[Reflection Agent]
    SM -->|High Confidence| FE[Feature Engineering Agent]
    
    REF --> FE
    
    FE --> ML[ML Evaluation Agent]
    ML --> REC[Recommendation Agent]
    
    REC -->|Borderline / Disagreement / Stuffing| REV[Review Agent]
    REC -->|High Confidence| EXP[Explanation Agent]
    
    REV --> EXP
    EXP --> END_NODE([END: Explainable Result])
    
    classDef agent fill:#e0e7ff,stroke:#4338ca,stroke-width:2px,color:#1e1b4b;
    classDef review fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f;
    classDef terminal fill:#f1f5f9,stroke:#475569,stroke-width:2px,color:#0f172a;
    
    class RP,ATS,SM,FE,ML,REC,EXP agent;
    class ER,REF,REV review;
    class START,FAIL,END_NODE terminal;
"""
