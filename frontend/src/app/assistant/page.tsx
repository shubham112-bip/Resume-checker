"use client";

import React, { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { EvaluationRunResponse, RunSummary } from "@/lib/types";
import { Card } from "@/components/Card";
import {
  MessageSquareCode,
  Send,
  Bot,
  User,
  ChevronDown,
  ChevronRight,
} from "lucide-react";

interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  generatedBy?: string;
  toolCalls?: any[];
  sources?: string[];
}

export default function AssistantPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [activeRun, setActiveRun] = useState<EvaluationRunResponse | null>(null);

  const [inputQuery, setInputQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "m_welcome",
      sender: "assistant",
      text: "Hello! I can answer questions about your evaluated candidates, compare qualifications, and locate specific resume evidence for your job requirements.",
      generatedBy: "grounded-agent",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});

  useEffect(() => {
    api.getRuns().then((rList) => {
      setRuns(rList);
      if (rList.length > 0) {
        setSelectedRunId(rList[0].runId);
      }
    });
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      api.getRun(selectedRunId).then((r) => setActiveRun(r)).catch(() => {});
    }
  }, [selectedRunId]);

  const handleSend = async (queryToSend?: string) => {
    const q = queryToSend || inputQuery;
    if (!q.trim()) return;

    const userMsg: Message = {
      id: `u_${Date.now()}`,
      sender: "user",
      text: q,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInputQuery("");
    setLoading(true);

    try {
      const resp = await api.recruiterChat(q, selectedRunId || undefined);
      const assistantMsg: Message = {
        id: `a_${Date.now()}`,
        sender: "assistant",
        text: resp.answer,
        generatedBy: resp.generatedBy,
        toolCalls: resp.toolCalls,
        sources: resp.sources,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errorMsg: Message = {
        id: `e_${Date.now()}`,
        sender: "assistant",
        text: `Error processing query: ${err.message}`,
        generatedBy: "error",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  // Dynamic suggestion chips based on active run candidates
  const topCands = activeRun?.ranking || [];
  const c1 = topCands[0]?.candidateName || "Candidate 1";
  const c2 = topCands[1]?.candidateName || "Candidate 2";
  const c3 = topCands[2]?.candidateName || "Candidate 3";

  const suggestions = [
    `Why was ${c1} shortlisted as the top candidate?`,
    `Compare ${c1} and ${c2}.`,
    `What skills or experience are missing from ${c3}?`,
    `Which candidate has the strongest React or backend experience?`,
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-200 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center space-x-2">
            <MessageSquareCode className="w-6 h-6 text-indigo-600" />
            <span>Recruiter AI Decision Support Assistant</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Answers natural language questions grounded strictly in stored evaluation evidence.
          </p>
        </div>

        {/* Run Selector */}
        {runs.length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-xs font-semibold text-slate-600">Run Context:</span>
            <select
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
              className="text-xs bg-white border border-slate-200 rounded-lg px-3 py-1.5 font-medium text-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            >
              {runs.map((r) => (
                <option key={r.runId} value={r.runId}>
                  {r.jobTitle} ({r.candidateCount} cands)
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Suggested Chips */}
      {topCands.length > 0 && (
        <div className="space-y-1.5">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
            Suggested Recruiter Questions:
          </span>
          <div className="flex flex-wrap gap-1.5">
            {suggestions.map((s, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(s)}
                className="px-3 py-1.5 rounded-full bg-white border border-slate-200 text-slate-700 hover:border-indigo-400 hover:text-indigo-700 text-xs transition shadow-sm"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Chat Messages Card */}
      <Card className="min-h-[460px] flex flex-col justify-between p-0">
        <div className="p-5 space-y-4 max-h-[500px] overflow-y-auto">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex items-start space-x-3 text-xs ${
                m.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {m.sender === "assistant" && (
                <div className="w-7 h-7 rounded-lg bg-indigo-600 text-white flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`p-4 rounded-2xl max-w-2xl leading-relaxed space-y-2 ${
                  m.sender === "user"
                    ? "bg-indigo-600 text-white rounded-tr-none shadow-sm"
                    : "bg-slate-50 border border-slate-200 text-slate-800 rounded-tl-none"
                }`}
              >
                <div className="whitespace-pre-wrap font-sans">{m.text}</div>

                {/* Tool calls collapsible (ReAct trace) */}
                {m.toolCalls && m.toolCalls.length > 0 && (
                  <div className="pt-2 border-t border-slate-200 text-[11px] font-mono">
                    <button
                      onClick={() =>
                        setExpandedTools((prev) => ({ ...prev, [m.id]: !prev[m.id] }))
                      }
                      className="text-indigo-600 font-semibold flex items-center space-x-1 hover:underline"
                    >
                      {expandedTools[m.id] ? (
                        <ChevronDown className="w-3.5 h-3.5" />
                      ) : (
                        <ChevronRight className="w-3.5 h-3.5" />
                      )}
                      <span>
                        Tool Invocations ({m.toolCalls.length} tool call
                        {m.toolCalls.length > 1 ? "s" : ""})
                      </span>
                    </button>

                    {expandedTools[m.id] && (
                      <div className="mt-2 space-y-1.5 p-2 rounded bg-white border border-slate-200">
                        {m.toolCalls.map((tc, idx) => (
                          <div key={idx} className="space-y-0.5">
                            <span className="text-indigo-700 font-bold block">{tc.tool}</span>
                            <span className="text-slate-600 text-[10px] block">
                              Arguments: {JSON.stringify(tc.arguments)}
                            </span>
                            <span className="text-slate-800 text-[10px] block">
                              Result: {tc.resultSummary}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Sources */}
                {m.sources && m.sources.length > 0 && (
                  <div className="text-[10px] text-slate-400 font-mono pt-1">
                    Grounded in: {m.sources.join(", ")}
                  </div>
                )}
              </div>

              {m.sender === "user" && (
                <div className="w-7 h-7 rounded-lg bg-slate-800 text-white flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex items-center space-x-2 text-xs text-slate-400 py-2">
              <div className="animate-spin w-4 h-4 border-2 border-indigo-600 border-t-transparent rounded-full" />
              <span>Querying evaluation evidence and generating answer...</span>
            </div>
          )}
        </div>

        {/* Input box */}
        <div className="p-4 border-t border-slate-200 bg-white">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="flex items-center space-x-2"
          >
            <input
              type="text"
              placeholder="Ask about candidate comparisons, missing skills, or ranking rationale..."
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              className="flex-1 text-xs px-4 py-3 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500 font-sans"
            />
            <button
              type="submit"
              disabled={loading || !inputQuery.trim()}
              className={`p-3 rounded-xl flex items-center justify-center transition shadow-sm ${
                loading || !inputQuery.trim()
                  ? "bg-slate-100 text-slate-300 cursor-not-allowed"
                  : "bg-indigo-600 text-white hover:bg-indigo-700"
              }`}
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </Card>
    </div>
  );
}
