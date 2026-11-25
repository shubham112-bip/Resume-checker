import React from "react";
import { Recommendation } from "@/lib/types";

interface RecommendationBadgeProps {
  recommendation: Recommendation;
  label?: string;
  size?: "sm" | "md" | "lg";
}

export function RecommendationBadge({
  recommendation,
  label,
  size = "md",
}: RecommendationBadgeProps) {
  let bgClass = "bg-slate-100 text-slate-700 border-slate-200";
  let dotClass = "bg-slate-400";
  let text = label || "Weak Match";

  switch (recommendation) {
    case "STRONG_SHORTLIST":
      bgClass = "bg-emerald-50 text-emerald-800 border-emerald-300";
      dotClass = "bg-emerald-500";
      text = label || "Strong Match";
      break;
    case "SHORTLIST":
      bgClass = "bg-teal-50 text-teal-800 border-teal-300";
      dotClass = "bg-teal-500";
      text = label || "Match";
      break;
    case "REVIEW":
      bgClass = "bg-amber-50 text-amber-800 border-amber-300";
      dotClass = "bg-amber-500";
      text = label || "Needs Review";
      break;
    case "WEAK_MATCH":
    default:
      bgClass = "bg-slate-100 text-slate-700 border-slate-300";
      dotClass = "bg-slate-400";
      text = label || "Weak Match";
      break;
  }

  const sizeClass =
    size === "sm"
      ? "px-2 py-0.5 text-[11px]"
      : size === "lg"
      ? "px-3.5 py-1.5 text-sm font-semibold"
      : "px-2.5 py-1 text-xs font-medium";

  return (
    <span
      className={`inline-flex items-center space-x-1.5 rounded-full border shadow-sm ${bgClass} ${sizeClass}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dotClass}`} />
      <span>{text}</span>
    </span>
  );
}
