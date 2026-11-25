"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Bot,
  History,
  ShieldCheck,
  BarChart3,
  Database,
  Network,
} from "lucide-react";

export function Sidebar() {
  const pathname = usePathname();

  const isLinkActive = (path: string) => {
    if (path === "/") {
      return pathname === "/";
    }
    return pathname.startsWith(path);
  };

  const navItemClass = (path: string) => {
    const active = isLinkActive(path);
    return `flex items-center gap-3 px-3.5 py-2.5 rounded-lg transition-all text-xs font-medium ${
      active
        ? "bg-slate-100 text-slate-900 font-semibold border-l-2 border-slate-900 shadow-xs"
        : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
    }`;
  };

  return (
    <aside className="fixed left-0 top-0 h-screen w-72 bg-white border-r border-slate-200 z-50 flex flex-col justify-between overflow-y-auto">
      <div className="flex flex-col">
        {/* Brand Header */}
        <div className="h-16 px-6 flex items-center justify-between border-b border-slate-200">
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-8 h-8 rounded-lg bg-slate-900 text-white flex items-center justify-center font-bold text-sm shadow-xs group-hover:bg-slate-800 transition-colors">
              TL
            </div>
            <div className="flex flex-col">
              <span className="font-headline-sm text-sm text-slate-900 tracking-tight leading-none font-bold">
                TalentLens
              </span>
              <span className="text-[11px] text-slate-500 font-medium mt-0.5">
                Screening Platform
              </span>
            </div>
          </Link>
        </div>

        {/* Primary Recruiter Navigation */}
        <div className="px-4 py-4">
          <div className="px-3 py-1 mb-1">
            <span className="text-[10px] uppercase text-slate-400 tracking-wider font-bold">
              Screening Workspace
            </span>
          </div>
          <nav className="flex flex-col gap-1">
            <Link href="/" className={navItemClass("/")}>
              <LayoutDashboard className="w-4 h-4 text-slate-500" />
              <span>Dashboard</span>
            </Link>
            <Link href="/history" className={navItemClass("/history")}>
              <History className="w-4 h-4 text-slate-500" />
              <span>Candidate Directory</span>
            </Link>
            <Link href="/assistant" className={navItemClass("/assistant")}>
              <Bot className="w-4 h-4 text-slate-500" />
              <span>Hiring Assistant</span>
            </Link>
          </nav>

          <div className="my-5 border-t border-slate-200" />

          {/* Secondary / Technical System Navigation */}
          <div className="px-3 py-1 mb-1">
            <span className="text-[10px] uppercase text-slate-400 tracking-wider font-bold">
              System & Auditing
            </span>
          </div>
          <nav className="flex flex-col gap-1">
            <Link href="/evaluation" className={navItemClass("/evaluation")}>
              <BarChart3 className="w-4 h-4 text-slate-500" />
              <span>Model Benchmarks</span>
            </Link>
            <Link href="/dataset" className={navItemClass("/dataset")}>
              <Database className="w-4 h-4 text-slate-500" />
              <span>Corpus Explorer</span>
            </Link>
            <Link href="/architecture" className={navItemClass("/architecture")}>
              <Network className="w-4 h-4 text-slate-500" />
              <span>System Diagnostics</span>
            </Link>
          </nav>
        </div>
      </div>

      {/* Fairness Guarantee Card */}
      <div className="p-4 border-t border-slate-200 bg-slate-50/60">
        <div className="p-3.5 rounded-xl bg-white border border-slate-200 shadow-xs space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5">
              <ShieldCheck className="w-4 h-4 text-emerald-600" />
              Fairness Safeguard
            </span>
            <span className="text-[10px] text-emerald-700 font-bold uppercase bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
              ACTIVE
            </span>
          </div>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            Protected attributes are excluded from candidate evaluation features.
          </p>
        </div>
      </div>
    </aside>
  );
}
