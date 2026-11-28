"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, Briefcase } from "lucide-react";
import { api } from "@/lib/api";
import { DemoJob } from "@/lib/types";

interface HeaderProps {
  currentJobTitle?: string;
  onSelectJob?: (demoId: string) => void;
}

export function Header({ currentJobTitle, onSelectJob }: HeaderProps) {
  const router = useRouter();
  const [demoJobs, setDemoJobs] = useState<DemoJob[]>([]);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  useEffect(() => {
    api.getDemoJobs().then(setDemoJobs).catch(() => { });
  }, []);

  return (
    <header className="fixed top-0 left-72 right-0 h-16 bg-white/95 backdrop-blur-md border-b border-slate-200 z-40 px-6 flex items-center justify-between gap-4">
      {/* Role / Requisition context */}
      <div className="flex items-center gap-4 flex-1 max-w-xl">
        <div className="relative">
          <button
            onClick={() => setDropdownOpen(!dropdownOpen)}
            className="flex items-center gap-2 bg-slate-50 hover:bg-slate-100 px-3.5 py-2 rounded-xl border border-slate-200 text-xs transition"
          >
            <Briefcase className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-slate-500 font-medium">Requisition:</span>
            <span className="text-slate-900 font-semibold max-w-[280px] truncate">
              {currentJobTitle || "Select Requisition..."}
            </span>
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 ml-0.5" />
          </button>

          {dropdownOpen && (
            <div className="absolute top-12 left-0 w-80 bg-white border border-slate-200 rounded-xl shadow-lg py-2 z-50">
              <div className="px-3 py-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Active Job Requisitions
              </div>
              {demoJobs.map((job) => (
                <button
                  key={job.demoId}
                  onClick={() => {
                    setDropdownOpen(false);
                    if (onSelectJob) {
                      onSelectJob(job.demoId);
                    } else {
                      router.push(`/?demo=${job.demoId}`);
                    }
                  }}
                  className="w-full text-left px-3 py-2 text-xs hover:bg-slate-50 flex flex-col"
                >
                  <span className="font-semibold text-slate-800 truncate">{job.title}</span>
                  <span className="text-[11px] text-slate-400">Code: {job.demoId}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-4 shrink-0">
        {/* Recruiter Final Authority Badge */}
        <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200 text-xs">
          <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" />
          <span className="font-medium text-[11px]">Decision Support</span>
        </div>

        <div className="h-5 w-px bg-slate-200" />

        {/* Recruiter Profile */}
        <div className="flex items-center gap-2.5 pl-1">
          <div className="w-8 h-8 rounded-full bg-slate-800 text-white font-semibold text-xs flex items-center justify-center border border-slate-200 shadow-xs">
            R
          </div>
          <div className="hidden md:flex flex-col text-left">
            <span className="text-xs font-semibold text-slate-900 leading-tight">
              Recruiter
            </span>
            <span className="text-[10px] text-slate-500 leading-tight">
              Talent Operations
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
