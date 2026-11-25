"use client";

import React, { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { GraphResponse } from "@/lib/types";
import { Card } from "@/components/Card";
import { MermaidViewer } from "@/components/MermaidViewer";
import {
  Network,
  BookOpen,
  Scale,
} from "lucide-react";

export default function ArchitecturePage() {
  const [graphData, setGraphData] = useState<GraphResponse | null>(null);
  const [config, setConfig] = useState<any>(null);

  useEffect(() => {
    api.getAgentGraph().then((g) => setGraphData(g)).catch(() => {});
    api.getConfig().then((c) => setConfig(c)).catch(() => {});
  }, []);

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center space-x-2">
          <Network className="w-6 h-6 text-indigo-600" />
          <span>System Diagnostics & Academic Architecture</span>
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Detailed academic mapping of AI concepts taught in college curricula to concrete implementations in this system.
        </p>
      </div>

      {/* 1. LangGraph Multi-Agent Architecture */}
      <Card
        title="LangGraph Multi-Agent State Graph Architecture"
        subtitle="Nodes, conditional edge branching, reflection loops, and error recovery transitions"
        icon={Network}
      >
        <div className="space-y-4">
          {graphData ? (
            <MermaidViewer chart={graphData.mermaid} />
          ) : (
            <div className="py-12 text-center text-xs text-slate-400">Loading state graph...</div>
          )}

          <div className="pt-2 border-t border-slate-100">
            <span className="font-semibold text-xs text-slate-700 block mb-2">
              Conditional Transitions Implemented in LangGraph:
            </span>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs font-mono">
              {graphData?.conditionalEdges.map((e, idx) => (
                <div key={idx} className="p-2 rounded bg-slate-50 border border-slate-200 text-slate-700">
                  {e}
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* 2. Transparent Scoring Formula */}
      <Card
        title="Transparent Scoring Framework"
        subtitle="Centrally configured mathematical formula ensuring explainable candidate-fit calculation"
        icon={Scale}
      >
        <div className="space-y-4 text-xs">
          <div className="p-3.5 rounded-xl bg-indigo-50 border border-indigo-200 font-mono text-indigo-950 font-medium">
            {config?.formula ||
              "Overall Score = 0.30 × Required Skill + 0.20 × Experience + 0.15 × Semantic Similarity + 0.10 × Project + 0.10 × ATS + 0.10 × Education + 0.05 × Quality"}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center font-mono">
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Req Technical Skill</span>
              <span className="text-base font-bold text-slate-900">30%</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Experience Match</span>
              <span className="text-base font-bold text-slate-900">20%</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Semantic Similarity</span>
              <span className="text-base font-bold text-slate-900">15%</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Project Relevance</span>
              <span className="text-base font-bold text-slate-900">10%</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">ATS Compatibility</span>
              <span className="text-base font-bold text-slate-900">10%</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Education & Degree</span>
              <span className="text-base font-bold text-slate-900">10%</span>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Formatting Quality</span>
              <span className="text-base font-bold text-slate-900">5%</span>
            </div>
            <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200">
              <span className="text-emerald-700 block text-[10px] font-bold">TOTAL SUM</span>
              <span className="text-base font-bold text-emerald-800">100%</span>
            </div>
          </div>
        </div>
      </Card>

      {/* 3. Comprehensive Syllabus Mapping */}
      <Card
        title="College AI Syllabus Concept Mapping"
        subtitle="Direct alignment with university MCA/BTech syllabus topics"
        icon={BookOpen}
      >
        <div className="space-y-4 text-xs divide-y divide-slate-100">
          {/* PEAS */}
          <div className="pt-3 first:pt-0 space-y-1">
            <span className="font-bold text-slate-900 text-sm">1. PEAS Framework</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>Performance:</strong> Candidate-job fit score (0-100), classification F1-score, shortlist recall, fairness.<br />
              <strong>Environment:</strong> Semi-structured candidate resumes (PDF/DOCX) and unstructured Job Descriptions.<br />
              <strong>Actuators:</strong> Ranking lists, recommendation bands, review flags, and structured evaluation reports.<br />
              <strong>Sensors:</strong> PyMuPDF / python-docx text extractors, sentence transformer vector embeddings, FAISS search results.
            </p>
          </div>

          {/* Rational & Agent Types */}
          <div className="pt-3 space-y-1">
            <span className="font-bold text-slate-900 text-sm">2. Rational Agents & Agent Types</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>Reactive Agent:</strong> ATS Agent applies direct deterministic rule-based checks on contact info and layout.<br />
              <strong>Deliberative Agent:</strong> Reflection Agent deliberates over implied skills (e.g. Django implying Python).<br />
              <strong>Learning Agent:</strong> Scikit-learn classification models (LR/RF/KNN) trained on labeled candidate outcomes.<br />
              <strong>Goal-Based Orchestration:</strong> LangGraph state graph coordinates sequential and conditional handoffs.
            </p>
          </div>

          {/* Natural Language Processing & Transformers */}
          <div className="pt-3 space-y-1">
            <span className="font-bold text-slate-900 text-sm">3. NLP, Tokens & Transformer Embeddings</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>Tokens:</strong> Resumes tokenized into subwords via HuggingFace WordPiece/BERT tokenizers.<br />
              <strong>Embeddings:</strong> Text passages encoded into 384-dimensional dense vectors using <code>BAAI/bge-small-en-v1.5</code>.<br />
              <strong>Vector Similarity:</strong> Dense embeddings enable exact semantic cosine similarity matching against candidate experience passages.
            </p>
          </div>

          {/* Vector Database & Semantic Retrieval */}
          <div className="pt-3 space-y-1">
            <span className="font-bold text-slate-900 text-sm">4. Vector Databases & Semantic Search</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>Chunking:</strong> Resumes segmented into semantic passages with section metadata (experience, projects, summary).<br />
              <strong>Indexing:</strong> FAISS <code>IndexFlatIP</code> performs exact inner product search over L2-normalized vectors (cosine similarity).<br />
              <strong>Semantic Match:</strong> Matches concepts like &ldquo;Express and JavaScript backend&rdquo; to &ldquo;Node.js&rdquo;.
            </p>
          </div>

          {/* ReAct & Tool Use */}
          <div className="pt-3 space-y-1">
            <span className="font-bold text-slate-900 text-sm">5. ReAct Paradigm, Tool Use & MCP</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>ReAct:</strong> Agents execute typed tools (<code>parse_resume</code>, <code>vector_search</code>, <code>predict_ml_fit</code>), observe structured returns, and transition states.<br />
              <strong>Action Trace:</strong> Visible timeline displays safe tool inputs and observations without exposing hidden chain-of-thought.<br />
              <strong>Model Context Protocol (MCP):</strong> Implemented in <code>app/mcp_server.py</code> to expose tools to external IDEs.
            </p>
          </div>

          {/* Memory */}
          <div className="pt-3 space-y-1">
            <span className="font-bold text-slate-900 text-sm">6. Agent Memory Architecture</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>Short-Term Memory:</strong> In-flight Pydantic <code>CandidateAgentState</code> within LangGraph during pipeline execution.<br />
              <strong>Long-Term Memory:</strong> SQLite storage of candidate profiles and job descriptions.<br />
              <strong>Episodic Memory:</strong> Historical log of candidate evaluation outcomes across past recruitment cycles.
            </p>
          </div>

          {/* AI Ethics & Bias Safeguards */}
          <div className="pt-3 space-y-1">
            <span className="font-bold text-slate-900 text-sm">7. AI Ethics, Bias Mitigation & Explainability</span>
            <p className="text-slate-600 leading-relaxed">
              <strong>Data Sanitization:</strong> Sensitive demographic attributes (gender, age, marital status, caste, religion, photo) are detected and stripped before analysis.<br />
              <strong>Explainability:</strong> Numerical scores are calculated mathematically from measurable features via calibrated statistical scoring and deterministic rule evaluations.<br />
              <strong>Human-in-the-Loop:</strong> Borderline scores, ML divergences, and keyword stuffing are automatically flagged for recruiter review.
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
}
