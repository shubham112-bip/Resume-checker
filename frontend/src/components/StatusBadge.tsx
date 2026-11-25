import React from "react";
import { EvidenceStatus } from "@/lib/types";

interface StatusBadgeProps {
  status: EvidenceStatus;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  let styleClass = "bg-slate-100 text-slate-600 border-slate-200";
  let label: string = status;

  switch (status) {
    case "MATCH":
      styleClass = "bg-emerald-50 text-emerald-700 border-emerald-200";
      label = "VERIFIED MATCH";
      break;
    case "SEMANTIC_MATCH":
      styleClass = "bg-sky-50 text-sky-700 border-sky-200";
      label = "SEMANTIC MATCH";
      break;
    case "CLAIMED_ONLY":
      styleClass = "bg-amber-50 text-amber-700 border-amber-200";
      label = "CLAIMED ONLY";
      break;
    case "INFERRED":
      styleClass = "bg-indigo-50 text-indigo-700 border-indigo-200";
      label = "INFERRED SKILL";
      break;
    case "UNCERTAIN":
      styleClass = "bg-yellow-50 text-yellow-800 border-yellow-200";
      label = "UNCERTAIN";
      break;
    case "NOT_FOUND":
      styleClass = "bg-rose-50 text-rose-700 border-rose-200";
      label = "NOT FOUND";
      break;
  }

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold tracking-wide uppercase border ${styleClass}`}
    >
      {label}
    </span>
  );
}
