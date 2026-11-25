"""Model Context Protocol (MCP) server for candidate evaluation tools."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.agents.tools import (
    tool_analyze_job,
    tool_evaluate_ats_rules,
    tool_parse_resume,
    tool_predict_ml_fit,
    tool_recall_memory,
    tool_search_resume_evidence,
)
from app.services.ml.predictor import get_cached_metrics
from mcp.server.mcpserver import MCPServer

logger = logging.getLogger(__name__)

# Initialize MCP Server
mcp = MCPServer(name="ai-resume-evaluator", version="1.0.0")


@mcp.tool()
async def parse_resume_document(file_path: str) -> str:
    """Parse a resume file (PDF, DOCX, TXT) and return structured candidate profile."""
    res = tool_parse_resume(file_path)
    return json.dumps(res, indent=2)


@mcp.tool()
async def analyze_job_posting(job_description_text: str) -> str:
    """Analyze a job description text and classify required and preferred skills."""
    res = tool_analyze_job(job_description_text)
    return json.dumps(res, indent=2)


@mcp.tool()
async def search_resume_passages(candidate_id: str, requirement: str, top_k: int = 3) -> str:
    """Search vector embeddings of a parsed resume for evidence passages matching a requirement."""
    res = tool_search_resume_evidence(candidate_id, requirement, top_k)
    return json.dumps(res, indent=2)


@mcp.tool()
async def run_ats_compatibility(candidate_id: str, job_id: str) -> str:
    """Run traditional ATS rule-based checks on a candidate against a job."""
    res = tool_evaluate_ats_rules(candidate_id, job_id)
    return json.dumps(res, indent=2)


@mcp.tool()
async def get_ml_evaluation_metrics() -> str:
    """Retrieve comparative evaluation metrics (accuracy, precision, recall, F1) for LR, RF, KNN."""
    metrics = get_cached_metrics()
    return json.dumps(metrics.model_dump(by_alias=True), indent=2)


def run_mcp_stdio() -> None:
    """Run the MCP server over standard I/O."""
    from mcp.server.stdio import stdio_server

    async def _main():
        async with stdio_server() as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    asyncio.run(_main())


if __name__ == "__main__":
    run_mcp_stdio()
