import React from "react";

interface ScoreBarProps {
  label: string;
  value: number; // 0 - 100
  weight?: number; // e.g. 0.30
  showValue?: boolean;
  size?: "sm" | "md" | "lg";
}

export function ScoreBar({
  label,
  value,
  weight,
  showValue = true,
  size = "md",
}: ScoreBarProps) {
  const clamped = Math.max(0, Math.min(100, value));

  let colorClass = "bg-rose-500";
  let textClass = "text-rose-700";
  if (clamped >= 85) {
    colorClass = "bg-emerald-500";
    textClass = "text-emerald-700";
  } else if (clamped >= 70) {
    colorClass = "bg-teal-500";
    textClass = "text-teal-700";
  } else if (clamped >= 55) {
    colorClass = "bg-amber-500";
    textClass = "text-amber-700";
  }

  const heightClass = size === "sm" ? "h-1.5" : size === "lg" ? "h-3" : "h-2";

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-1 text-xs">
        <span className="font-medium text-slate-700 truncate pr-2">
          {label}
          {weight !== undefined && (
            <span className="text-slate-400 font-normal ml-1.5 text-[11px]">
              ({Math.round(weight * 100)}%)
            </span>
          )}
        </span>
        {showValue && (
          <span className={`font-semibold font-mono ${textClass}`}>
            {clamped.toFixed(1)}/100
          </span>
        )}
      </div>
      <div className={`w-full bg-slate-100 rounded-full overflow-hidden ${heightClass}`}>
        <div
          className={`${colorClass} ${heightClass} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}
