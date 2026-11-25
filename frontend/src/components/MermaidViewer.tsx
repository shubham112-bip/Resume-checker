"use client";

import React, { useEffect, useRef, useState } from "react";

interface MermaidViewerProps {
  chart: string;
}

export function MermaidViewer({ chart }: MermaidViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [svgContent, setSvgContent] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    import("mermaid").then((mermaidModule) => {
      const mermaid = mermaidModule.default;
      mermaid.initialize({
        startOnLoad: false,
        theme: "neutral",
        securityLevel: "loose",
        flowchart: {
          curve: "basis",
          padding: 15,
        },
      });

      const uniqueId = `mermaid-${Math.random().toString(36).substring(2, 9)}`;
      mermaid
        .render(uniqueId, chart)
        .then(({ svg }) => {
          if (isMounted) {
            setSvgContent(svg);
            setError(null);
          }
        })
        .catch((err) => {
          if (isMounted) {
            console.error("Mermaid render error:", err);
            setError("Diagram rendering error. Showing raw diagram definition below.");
          }
        });
    });

    return () => {
      isMounted = false;
    };
  }, [chart]);

  if (error) {
    return (
      <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg text-xs font-mono text-slate-700 whitespace-pre overflow-x-auto">
        <div className="text-amber-600 mb-2 font-sans font-medium">{error}</div>
        {chart}
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="overflow-x-auto flex justify-center py-4 bg-slate-50/50 rounded-xl border border-slate-100"
      dangerouslySetInnerHTML={{ __html: svgContent }}
    />
  );
}
