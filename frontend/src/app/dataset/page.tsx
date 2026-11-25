"use client";

import React, { useEffect, useState } from "react";
import { api } from "@/lib/api";
import {
  DatasetPairSummary,
  DatasetSummaryResponse,
  ResumeQualityUploadResponse,
} from "@/lib/types";
import { Card } from "@/components/Card";
import {
  Database,
  FileText,
  CheckCircle2,
  AlertCircle,
  Upload,
  Sparkles,
  ShieldAlert,
  Layers,
  Search,
  Award,
} from "lucide-react";

export default function DatasetPage() {
  const [summary, setSummary] = useState<DatasetSummaryResponse | null>(null);
  const [pairs, setPairs] = useState<DatasetPairSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedJdFilter, setSelectedJdFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");

  // Standalone quality test state
  const [qualityFile, setQualityFile] = useState<File | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityResult, setQualityResult] = useState<ResumeQualityUploadResponse | null>(null);
  const [qualityError, setQualityError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.getDatasetSummary(), api.getDatasetPairs(100)])
      .then(([s, p]) => {
        setSummary(s);
        setPairs(p);
      })
      .catch((err) => console.error("Error loading dataset data:", err))
      .finally(() => setLoading(false));
  }, []);

  const handleUploadQuality = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!qualityFile) return;

    setQualityLoading(true);
    setQualityError(null);
    setQualityResult(null);

    try {
      const res = await api.evaluateResumeQuality(qualityFile);
      setQualityResult(res);
    } catch (err: any) {
      setQualityError(err.message || "Failed to evaluate resume quality.");
    } finally {
      setQualityLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="py-20 text-center text-slate-500 text-sm">
        <div className="animate-spin w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full mx-auto mb-3" />
        Loading benchmark dataset and quality metrics...
      </div>
    );
  }

  const filteredPairs = pairs.filter((p) => {
    const matchesJd = selectedJdFilter === "ALL" || p.jobTitle.toLowerCase().includes(selectedJdFilter.toLowerCase());
    const matchesSearch =
      searchQuery === "" ||
      p.candidateName.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.jobTitle.toLowerCase().includes(searchQuery.toLowerCase()) ||
      p.label.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesJd && matchesSearch;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="pb-4 border-b border-slate-200">
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center space-x-2">
          <Database className="w-6 h-6 text-slate-800" />
          <span>Benchmark Resume Corpus & Document Quality Analysis</span>
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Ingestion pipeline and rule-assisted development benchmark across 76 candidates (including 65 uploaded DOCX resumes) and 12 specialized Job Descriptions (912 Resume-JD pairs).
        </p>
      </div>

      {/* Academic Integrity Notice Banner */}
      <div className="p-4 rounded-xl bg-indigo-50 border border-indigo-200 text-indigo-950 text-xs flex items-start space-x-3">
        <ShieldAlert className="w-5 h-5 text-indigo-600 flex-shrink-0 mt-0.5" />
        <div>
          <span className="font-bold block mb-0.5">Academic Honesty & Methodological Integrity:</span>
          <p className="leading-relaxed">
            {summary?.notice ||
              "All pairs are evaluated using deterministic, rule-assisted labels without fabricated human evaluations. To avoid data leakage, ML models use Stratified Grouped K-Fold splitting by candidate ID so test resumes are never seen during training."}
          </p>
        </div>
      </div>

      {/* Core Principle Callout */}
      <div className="p-4 rounded-xl bg-slate-900 text-slate-100 text-xs">
        <div className="flex items-center space-x-2 font-bold text-emerald-400 mb-1">
          <Sparkles className="w-4 h-4" />
          <span>Core Viva Defense Concept: Resume Quality ≠ Candidate-Job Fit</span>
        </div>
        <p className="leading-relaxed text-slate-300">
          A candidate resume can achieve a high <strong>Resume Quality Score</strong> (e.g., 86/100) due to clear formatting, quantified achievements, and strong action verbs, yet have a low <strong>Candidate Fit Score</strong> (e.g., 45/100) if their tech stack does not match the target job description. Conversely, a plain resume can be a perfect fit for a specific role.
        </p>
      </div>

      {/* Dataset Statistics Counters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Sample Resumes</span>
          <span className="text-2xl font-black font-mono text-slate-900 mt-1 block">
            {summary ? summary.totalResumes : "—"}
          </span>
          <span className="text-[11px] text-slate-500 mt-0.5 block">PDF, DOCX, and TXT files</span>
        </div>

        <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Job Descriptions</span>
          <span className="text-2xl font-black font-mono text-indigo-600 mt-1 block">
            {summary ? summary.totalJobDescriptions : "—"}
          </span>
          <span className="text-[11px] text-slate-500 mt-0.5 block">Full Stack, Backend, DS, Cloud, etc.</span>
        </div>

        <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Evaluated Pairs</span>
          <span className="text-2xl font-black font-mono text-emerald-600 mt-1 block">
            {summary ? summary.totalPairs : "—"}
          </span>
          <span className="text-[11px] text-slate-500 mt-0.5 block">{summary?.totalResumes || 76} candidates × {summary?.totalJobDescriptions || 12} roles</span>
        </div>

        <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Label Source</span>
          <span className="text-2xl font-black font-mono text-amber-600 mt-1 block">RULE_BASED</span>
          <span className="text-[11px] text-slate-500 mt-0.5 block">Deterministic Rule-Assisted</span>
        </div>
      </div>

      {/* Ingested Resumes Table */}
      <Card
        title="Ingested Real Sample Resumes"
        subtitle="Parsed into normalized JSON schema with independent 6-dimensional Resume Quality Evaluation"
        icon={FileText}
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-500 font-semibold">
                <th className="py-2.5 px-3">Candidate</th>
                <th className="py-2.5 px-3">Format</th>
                <th className="py-2.5 px-3 text-right">Words</th>
                <th className="py-2.5 px-3 text-center">Quality Grade</th>
                <th className="py-2.5 px-3 text-right">Quality Score</th>
                <th className="py-2.5 px-3 text-center">Action Verbs</th>
                <th className="py-2.5 px-3 text-center">Quantified Metrics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {summary?.resumes.map((r) => (
                <tr key={r.resumeId} className="hover:bg-slate-50/70 transition">
                  <td className="py-2.5 px-3 font-semibold text-slate-900">
                    {r.candidateName}
                  </td>
                  <td className="py-2.5 px-3">
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-100 text-slate-700 font-bold border border-slate-200">
                      {r.fileType}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                    {r.wordCount}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <span
                      className={`px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                        r.qualityGrade === "Excellent"
                          ? "bg-emerald-100 text-emerald-800"
                          : r.qualityGrade === "Good"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-amber-100 text-amber-800"
                      }`}
                    >
                      {r.qualityGrade}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono font-bold text-slate-900">
                    {r.qualityScore.toFixed(1)} / 100
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono text-indigo-600 font-semibold">
                    {r.actionVerbsCount} verbs
                  </td>
                  <td className="py-2.5 px-3 text-center font-mono text-slate-600">
                    {r.quantifiedBulletsCount} metrics
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Standalone Resume Quality Checker Tool */}
      <Card
        title="Standalone Objective Resume Quality Tester"
        subtitle="Evaluate any resume (PDF, DOCX, TXT) for ATS readability, action verbs, quantified accomplishments, and structure"
        icon={Award}
      >
        <form onSubmit={handleUploadQuality} className="space-y-4">
          <div className="flex flex-col sm:flex-row items-center gap-3">
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setQualityFile(e.target.files?.[0] || null)}
              className="text-xs text-slate-600 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer w-full sm:w-auto"
            />
            <button
              type="submit"
              disabled={!qualityFile || qualityLoading}
              className="w-full sm:w-auto px-4 py-2 bg-indigo-600 text-white rounded-lg text-xs font-semibold hover:bg-indigo-700 transition disabled:opacity-50 flex items-center justify-center space-x-1.5"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>{qualityLoading ? "Evaluating Quality..." : "Evaluate Quality"}</span>
            </button>
          </div>

          {qualityError && (
            <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs">
              {qualityError}
            </div>
          )}

          {qualityResult && (
            <div className="mt-4 p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-4">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 pb-3 border-b border-slate-200">
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    {qualityResult.candidateName} — {qualityResult.fileName}
                  </h3>
                  <p className="text-xs text-slate-500 font-mono mt-0.5">
                    Grade: <span className="font-bold text-indigo-600">{qualityResult.evaluation.qualityGrade}</span>
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-2xl font-black font-mono text-slate-900">
                    {qualityResult.evaluation.overallQualityScore.toFixed(1)}
                  </span>
                  <span className="text-slate-400 text-xs"> / 100</span>
                </div>
              </div>

              {/* 6 Dimensions Breakdown */}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {qualityResult.evaluation.dimensions.map((dim) => (
                  <div key={dim.name} className="p-2.5 bg-white rounded-lg border border-slate-200 text-xs">
                    <div className="flex justify-between items-center text-[11px] font-semibold text-slate-700">
                      <span>{dim.name}</span>
                      <span className="font-mono text-indigo-600">{Math.round(dim.score * 100)}%</span>
                    </div>
                    <div className="w-full bg-slate-100 rounded-full h-1.5 mt-1.5 overflow-hidden">
                      <div
                        className="bg-indigo-600 h-1.5 rounded-full"
                        style={{ width: `${Math.min(100, Math.round(dim.score * 100))}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Action Verbs Found */}
              {qualityResult.evaluation.actionVerbsFound.length > 0 && (
                <div>
                  <span className="text-[11px] font-semibold text-slate-600 uppercase tracking-wider block mb-1.5">
                    Action Verbs Identified ({qualityResult.evaluation.actionVerbsFound.length})
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {qualityResult.evaluation.actionVerbsFound.map((v) => (
                      <span
                        key={v}
                        className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-50 text-indigo-700 border border-indigo-200 font-semibold"
                      >
                        {v}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Strengths & Improvement */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200 text-emerald-900 space-y-1">
                  <span className="font-bold flex items-center space-x-1">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Quality Strengths</span>
                  </span>
                  <ul className="list-disc list-inside space-y-0.5 text-[11px] text-emerald-800">
                    {qualityResult.evaluation.strengths.map((s, idx) => (
                      <li key={idx}>{s}</li>
                    ))}
                  </ul>
                </div>

                <div className="p-3 bg-amber-50 rounded-lg border border-amber-200 text-amber-900 space-y-1">
                  <span className="font-bold flex items-center space-x-1">
                    <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                    <span>Improvement Opportunities</span>
                  </span>
                  <ul className="list-disc list-inside space-y-0.5 text-[11px] text-amber-800">
                    {qualityResult.evaluation.improvementAreas.map((ia, idx) => (
                      <li key={idx}>{ia}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </form>
      </Card>

      {/* Resume-JD Labeled Pairs Explorer */}
      <Card
        title={`Resume-JD Evaluation Pairs Explorer (N=${summary?.totalPairs || 912})`}
        subtitle="Examine how the same candidate produces high fit for their target role and lower fit for non-target roles"
        icon={Layers}
      >
        {/* Controls */}
        <div className="flex flex-col sm:flex-row gap-3 mb-4">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search by candidate, role or label..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-3 py-1.5 rounded-lg border border-slate-200 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          <select
            value={selectedJdFilter}
            onChange={(e) => setSelectedJdFilter(e.target.value)}
            className="py-1.5 px-3 rounded-lg border border-slate-200 text-xs focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white"
          >
            <option value="ALL">All Roles ({pairs.length} pairs)</option>
            {summary?.jobDescriptions.map((jd) => (
              <option key={jd.jdId} value={jd.title}>
                {jd.title}
              </option>
            ))}
          </select>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-slate-500 font-semibold">
                <th className="py-2.5 px-3">Candidate</th>
                <th className="py-2.5 px-3">Target Job Description</th>
                <th className="py-2.5 px-3 text-right">Job Fit Score</th>
                <th className="py-2.5 px-3 text-right">Quality Score</th>
                <th className="py-2.5 px-3 text-center">Rule Label</th>
                <th className="py-2.5 px-3 text-center">ML Target</th>
                <th className="py-2.5 px-3 text-center">Label Source</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredPairs.map((p, idx) => (
                <tr key={`${p.resumeId}_${p.jdId}_${idx}`} className="hover:bg-slate-50/70 transition">
                  <td className="py-2 px-3 font-semibold text-slate-900">{p.candidateName}</td>
                  <td className="py-2 px-3 text-slate-700">{p.jobTitle}</td>
                  <td className="py-2 px-3 text-right font-mono font-bold text-indigo-600">
                    {p.overallCandidateFitScore.toFixed(1)}
                  </td>
                  <td className="py-2 px-3 text-right font-mono text-slate-600">
                    {p.resumeQualityScore.toFixed(1)}
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                        p.label === "STRONG_MATCH" || p.label === "MATCH"
                          ? "bg-emerald-100 text-emerald-800"
                          : p.label === "REVIEW"
                          ? "bg-blue-100 text-blue-800"
                          : "bg-slate-100 text-slate-700"
                      }`}
                    >
                      {p.label}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-center font-mono text-[11px] text-slate-600">
                    {p.classificationLabel}
                  </td>
                  <td className="py-2 px-3 text-center">
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-amber-50 text-amber-800 border border-amber-200">
                      {p.labelSource}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
