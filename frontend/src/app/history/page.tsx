"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import { CandidateSummary, EpisodeResponse } from "@/lib/types";
import { Card } from "@/components/Card";
import { Users, History, Calendar, ArrowRight } from "lucide-react";

export default function HistoryPage() {
  const [candidates, setCandidates] = useState<CandidateSummary[]>([]);
  const [episodes, setEpisodes] = useState<EpisodeResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.getCandidates(), api.getEpisodes()])
      .then(([c, e]) => {
        setCandidates(c);
        setEpisodes(e);
      })
      .catch((err) => console.error("Error loading history:", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="py-20 text-center text-slate-500 text-xs">
        <div className="animate-spin w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full mx-auto mb-3" />
        Loading candidate history...
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center space-x-2">
          <History className="w-6 h-6 text-slate-800" />
          <span>Candidate Directory & Screening History</span>
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Candidate profiles, past evaluations, and screening runs across recruiting requisitions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Candidates Directory */}
        <Card
          title="Candidate Profiles"
          subtitle={`${candidates.length} candidate(s) stored`}
          icon={Users}
        >
          {candidates.length > 0 ? (
            <div className="divide-y divide-slate-100 border border-slate-200 rounded-lg overflow-hidden text-xs">
              {candidates.map((c) => (
                <div
                  key={c.candidateId}
                  className="p-3 flex items-center justify-between hover:bg-slate-50 transition"
                >
                  <div>
                    <span className="font-semibold text-slate-900 block">{c.candidateName}</span>
                    <span className="text-[11px] text-slate-400">
                      File: {c.fileName}
                    </span>
                  </div>

                  <div className="flex items-center space-x-3">
                    {c.latestEvaluationId && (
                      <Link
                        href={`/evaluations/${c.latestEvaluationId}`}
                        className="text-slate-800 hover:text-slate-950 font-medium flex items-center space-x-1"
                      >
                        <span>View Evaluation</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400">No candidates recorded yet.</p>
          )}
        </Card>

        {/* Screening Timeline */}
        <Card
          title="Recent Screening Timeline"
          subtitle="Chronological log of candidate evaluations and hiring determinations"
          icon={Calendar}
        >
          {episodes.length > 0 ? (
            <div className="space-y-3 text-xs">
              {episodes.map((ep) => (
                <div
                  key={ep.episodeId}
                  className="p-3 rounded-lg border border-slate-200 bg-slate-50 space-y-1"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-900">
                      {ep.candidateName} &rarr; {ep.jobTitle}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {new Date(ep.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-slate-600">{ep.summary}</p>
                  {ep.overallScore !== null && (
                    <div className="text-[11px] text-slate-800 font-bold pt-0.5">
                      Score: {ep.overallScore.toFixed(0)} / 100
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400">No screening evaluations recorded yet.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
