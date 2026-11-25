"use client";

import React from "react";
import { Loader2 } from "lucide-react";

interface PipelineProgressProps {
  active: boolean;
}

export function AgentPipelineProgress({ active }: PipelineProgressProps) {
  if (!active) return null;

  return (
    <div className="bg-slate-900 text-white rounded-2xl p-6 shadow-md my-6 border border-slate-800">
      <div className="flex items-center space-x-3">
        <div className="p-2.5 bg-slate-800 rounded-xl text-sky-400">
          <Loader2 className="w-5 h-5 animate-spin" />
        </div>
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-sm tracking-tight text-white">
              Screening Candidates Against Requisition Requirements
            </h4>
            <span className="text-[10px] uppercase font-bold text-sky-400 bg-sky-950/60 px-2 py-0.5 rounded border border-sky-800">
              Processing
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Ingesting candidate resumes, matching qualifications against role requirements, and compiling evidence-based evaluations...
          </p>
        </div>
      </div>
    </div>
  );
}
