"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { api } from "@/lib/api";
import { HealthResponse } from "@/lib/types";
import {
  Brain,
  BarChart3,
  MessageSquareCode,
  Network,
  Users,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Sparkles,
  Database,
} from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api
      .getHealth()
      .then((data) => setHealth(data))
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, []);

  const navLinks = [
    { href: "/", label: "Dashboard", icon: Brain },
    { href: "/dataset", label: "Dataset & Quality", icon: Database },
    { href: "/evaluation", label: "Model Evaluation", icon: BarChart3 },
    { href: "/assistant", label: "Recruiter Assistant", icon: MessageSquareCode },
    { href: "/architecture", label: "Architecture", icon: Network },
    { href: "/history", label: "History & Memory", icon: Users },
  ];

  return (
    <header className="border-b border-slate-200 bg-white sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-2.5">
              <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-200">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <span className="font-bold text-lg text-slate-900 tracking-tight block leading-tight">
                  TalentGraph <span className="text-indigo-600">AI</span>
                </span>
                <span className="text-[10px] font-medium text-slate-500 uppercase tracking-wider block">
                  Agentic Resume Evaluator
                </span>
              </div>
            </Link>

            <nav className="hidden md:flex space-x-1 pl-6">
              {navLinks.map((link) => {
                const Icon = link.icon;
                const isActive = pathname === link.href || (link.href !== "/" && pathname.startsWith(link.href));
                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`flex items-center space-x-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                      isActive
                        ? "bg-indigo-50 text-indigo-700"
                        : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? "text-indigo-600" : "text-slate-400"}`} />
                    <span>{link.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          {/* Health and Model status pills */}
          <div className="flex items-center space-x-2 text-xs">
            {loading ? (
              <span className="text-slate-400 animate-pulse">Checking backend...</span>
            ) : health ? (
              <div className="flex items-center space-x-2">
                <div className="hidden lg:flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                  <Cpu className="w-3.5 h-3.5 text-slate-500" />
                  <span className="font-mono text-[11px]">{health.selectedModel || "ML Ready"}</span>
                </div>

                <div
                  className={`flex items-center space-x-1.5 px-2.5 py-1 rounded-full border text-[11px] font-medium ${
                    health.geminiEnabled
                      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                      : "bg-amber-50 text-amber-700 border-amber-200"
                  }`}
                  title={health.geminiEnabled ? `Gemini Active (${health.geminiModel})` : "Grounded Fallback Engine Active"}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{health.geminiEnabled ? "Gemini Live" : "Grounded Engine"}</span>
                </div>

                <div className="flex items-center space-x-1 px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                  <span className="font-medium text-[11px]">API Online</span>
                </div>
              </div>
            ) : (
              <div className="flex items-center space-x-1 px-2.5 py-1 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                <AlertCircle className="w-3.5 h-3.5 text-rose-600" />
                <span className="font-medium">Backend Offline</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
