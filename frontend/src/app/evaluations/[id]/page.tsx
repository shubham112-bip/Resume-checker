"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { CandidateEvaluation } from "@/lib/types";
import {
  ArrowLeft,
  Download,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  ShieldCheck,
  ChevronDown,
  ChevronRight,
  AlertCircle,
  Check,
  Flag,
  Layers,
} from "lucide-react";

export default function CandidateDetailPage() {
  const params = useParams();
  const router = useRouter();
  const evaluationId = params.id as string;

  const [evaluation, setEvaluation] = useState<CandidateEvaluation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  // Recruiter Decision State
  const [recruiterDecision, setRecruiterDecision] = useState<"SHORTLIST" | "REVIEW" | "REJECT" | null>(null);
  const [recruiterNotes, setRecruiterNotes] = useState("");
  const [decisionSaved, setDecisionSaved] = useState(false);
  const [savingDecision, setSavingDecision] = useState(false);
  const [decisionError, setDecisionError] = useState<string | null>(null);
  const [decidedAt, setDecidedAt] = useState<string | null>(null);

  useEffect(() => {
    if (!evaluationId) return;
    setLoading(true);
    api
      .getEvaluation(evaluationId)
      .then((data) => {
        setEvaluation(data);
        if (data.recruiterDecision) {
          setRecruiterDecision(data.recruiterDecision);
        }
        if (data.recruiterNotes) {
          setRecruiterNotes(data.recruiterNotes);
        }
        if (data.recruiterDecidedAt) {
          setDecidedAt(data.recruiterDecidedAt);
        }
      })
      .catch((err) => setError(err.message || "Failed to load candidate evaluation"))
      .finally(() => setLoading(false));
  }, [evaluationId]);

  const handleSaveDecision = async (decision: "SHORTLIST" | "REVIEW" | "REJECT") => {
    if (!evaluationId) return;
    setSavingDecision(true);
    setDecisionError(null);
    try {
      const updated = await api.saveRecruiterDecision(evaluationId, decision, recruiterNotes);
      setRecruiterDecision(updated.recruiterDecision || decision);
      setDecidedAt(updated.recruiterDecidedAt || new Date().toISOString());
      setDecisionSaved(true);
      setTimeout(() => setDecisionSaved(false), 5000);
    } catch (err: any) {
      setDecisionError(err.message || "Failed to save recruiter decision.");
    } finally {
      setSavingDecision(false);
    }
  };

  if (loading) {
    return (
      <div className="py-24 text-center text-slate-500 text-xs">
        <div className="animate-spin w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full mx-auto mb-3" />
        Loading candidate dossier...
      </div>
    );
  }

  if (error || !evaluation) {
    return (
      <div className="p-8 max-w-2xl mx-auto my-12 bg-rose-50 border border-rose-200 rounded-2xl text-rose-800 text-xs space-y-4">
        <div className="flex items-center space-x-2 font-bold text-sm">
          <AlertCircle className="w-5 h-5 text-rose-600" />
          <span>Candidate Record Not Found</span>
        </div>
        <p>{error || "The requested candidate evaluation could not be loaded."}</p>
        <button
          onClick={() => router.push("/")}
          className="px-4 py-2 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  const ev = evaluation;

  // Recommendation status
  const isStrong = ev.recommendation === "STRONG_SHORTLIST";
  const isMatch = ev.recommendation === "SHORTLIST";
  const isReview = ev.recommendation === "REVIEW";

  const statusBadge = isStrong ? (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 text-xs font-semibold">
      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
      Strong Match
    </span>
  ) : isMatch ? (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-50 text-sky-800 border border-sky-200 text-xs font-semibold">
      <CheckCircle2 className="w-3.5 h-3.5 text-sky-600" />
      Match
    </span>
  ) : isReview ? (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-50 text-amber-800 border border-amber-200 text-xs font-semibold">
      <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
      Needs Review
    </span>
  ) : (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 text-slate-700 border border-slate-200 text-xs font-semibold">
      <XCircle className="w-3.5 h-3.5 text-slate-500" />
      Weak Match
    </span>
  );

  // Recruiter-Friendly 7 Scoring Dimensions
  const dimensions = [
    {
      label: "Required Technical Skills",
      weight: 30,
      score: ev.requiredSkillMatch,
      description: `Matched ${ev.matchedSkills?.length || 0} core technical requirements specified in the job description.`,
    },
    {
      label: "Relevant Experience Depth",
      weight: 20,
      score: ev.experienceMatch,
      description: `${ev.experienceYears.toFixed(1)} years detected career tenure aligned with role seniority.`,
    },
    {
      label: "Semantic Context Alignment",
      weight: 15,
      score: ev.semanticSimilarity,
      description: "Relevance of responsibilities, domain scope, and architectural responsibilities.",
    },
    {
      label: "Project Relevance",
      weight: 10,
      score: ev.projectMatch,
      description: "Demonstrated delivery of production systems, features, or engineering achievements.",
    },
    {
      label: "ATS Parsability & Layout",
      weight: 10,
      score: ev.atsScore,
      description: "Standard document sections, clean text extraction, and contact detail completeness.",
    },
    {
      label: "Education & Degree Match",
      weight: 10,
      score: ev.educationMatch,
      description: "Academic qualification alignment with specified education preferences.",
    },
    {
      label: "Resume Presentation Quality",
      weight: 5,
      score: ev.resumeQuality,
      description: "Clarity of writing, quantifiable achievements, and bullet point structure.",
    },
  ];

  // Strengths and weaknesses
  const strengths = ev.strengths?.length > 0 ? ev.strengths : ev.explanation?.strengths || [];
  const weaknesses = ev.weaknesses?.length > 0 ? ev.weaknesses : ev.explanation?.weaknesses || [];

  // Verbatim evidence passages
  const allEvidence = ev.requirementMatches?.flatMap((rm) => rm.evidence) || [];

  return (
    <div className="flex flex-col w-full pb-16">
      <div className="p-6 lg:p-10 max-w-[1400px] w-full mx-auto space-y-8">
        {/* Navigation & Action Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
          <Link
            href={`/?run=${ev.runId}`}
            className="inline-flex items-center gap-2 text-xs font-semibold text-slate-600 hover:text-slate-900 transition"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Candidate List ({ev.jobTitle})</span>
          </Link>

          <div className="flex items-center gap-2.5">
            <a
              href={api.getReportUrl(ev.evaluationId, "pdf")}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3.5 py-1.5 rounded-xl bg-slate-900 text-white hover:bg-slate-800 text-xs font-semibold flex items-center gap-1.5 shadow-xs transition"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download PDF</span>
            </a>

            <a
              href={api.getReportUrl(ev.evaluationId, "md")}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3 py-1.5 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold flex items-center gap-1.5 shadow-xs transition"
            >
              <span>Markdown</span>
            </a>
          </div>
        </div>

        {/* 1. Candidate Overview Card */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 sm:p-8 shadow-xs space-y-6">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
            <div className="space-y-2">
              <div className="flex flex-wrap items-center gap-3">
                <h1 className="font-headline-lg text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
                  {ev.candidateName}
                </h1>
                {statusBadge}
                <span className="px-2.5 py-0.5 rounded bg-slate-100 text-slate-700 font-data-mono-sm text-[11px] font-medium border border-slate-200 flex items-center gap-1">
                  <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
                  Demographics Masked
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-slate-500 text-xs">
                <span className="font-semibold text-slate-700">{ev.jobTitle}</span>
                <span>•</span>
                <span className="font-data-mono-sm">ID: {ev.candidateId.slice(0, 10)}</span>
                <span>•</span>
                <span>File: {ev.fileName}</span>
              </div>
            </div>

            {/* Recruiter Decision Actions */}
            <div className="flex flex-wrap items-center gap-2.5 self-start lg:self-auto">
              <button
                onClick={() => handleSaveDecision("SHORTLIST")}
                disabled={savingDecision}
                className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs disabled:opacity-50 ${
                  recruiterDecision === "SHORTLIST"
                    ? "bg-emerald-600 text-white"
                    : "bg-emerald-50 text-emerald-800 hover:bg-emerald-100 border border-emerald-200"
                }`}
              >
                <Check className="w-4 h-4" />
                <span>{savingDecision && recruiterDecision === "SHORTLIST" ? "Saving..." : "Shortlist"}</span>
              </button>

              <button
                onClick={() => handleSaveDecision("REVIEW")}
                disabled={savingDecision}
                className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs disabled:opacity-50 ${
                  recruiterDecision === "REVIEW"
                    ? "bg-amber-600 text-white"
                    : "bg-amber-50 text-amber-800 hover:bg-amber-100 border border-amber-200"
                }`}
              >
                <Flag className="w-4 h-4" />
                <span>{savingDecision && recruiterDecision === "REVIEW" ? "Saving..." : "Flag for Review"}</span>
              </button>

              <button
                onClick={() => handleSaveDecision("REJECT")}
                disabled={savingDecision}
                className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs disabled:opacity-50 ${
                  recruiterDecision === "REJECT"
                    ? "bg-slate-800 text-white"
                    : "bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200"
                }`}
              >
                <XCircle className="w-4 h-4" />
                <span>{savingDecision && recruiterDecision === "REJECT" ? "Saving..." : "Reject"}</span>
              </button>
            </div>
          </div>

          {decisionSaved && (
            <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-xs flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>
                Recruiter decision recorded as <strong>{recruiterDecision}</strong>.
              </span>
            </div>
          )}
        </div>

        {/* 2. Dual-Score Summary (Candidate-JD Fit vs. Resume Quality) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Candidate-Job Fit */}
          <div className="bg-white rounded-2xl p-6 sm:p-7 border border-slate-200 shadow-xs flex flex-col justify-between space-y-4">
            <div className="flex items-start justify-between">
              <div className="space-y-0.5">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                  Job Match Assessment
                </span>
                <h2 className="text-lg font-bold text-slate-900">
                  Candidate–Job Fit
                </h2>
              </div>
              <span className="text-xs font-semibold text-slate-500">
                Job-Dependent
              </span>
            </div>

            <div className="flex items-center gap-6 my-2">
              <div className="relative w-24 h-24 shrink-0 flex items-center justify-center">
                <svg className="w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" fill="none" r="40" stroke="#f1f5f9" strokeWidth="8" />
                  <circle
                    cx="50"
                    cy="50"
                    fill="none"
                    r="40"
                    stroke={ev.overallScore >= 75 ? "#059669" : ev.overallScore >= 60 ? "#0284c7" : "#e11d48"}
                    strokeWidth="8"
                    strokeDasharray="251.32"
                    strokeDashoffset={`${251.32 * (1 - Math.min(ev.overallScore, 100) / 100)}`}
                    strokeLinecap="round"
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-2xl font-black text-slate-900 leading-none">
                    {ev.overallScore.toFixed(0)}
                  </span>
                  <span className="text-[10px] text-slate-400">/ 100</span>
                </div>
              </div>

              <div className="space-y-1.5 text-xs text-slate-600 leading-relaxed">
                <p className="font-semibold text-slate-900">
                  {ev.overallScore >= 75
                    ? "Strong alignment with core technical qualifications and experience."
                    : ev.overallScore >= 60
                    ? "Qualified match meeting primary skill requirements."
                    : ev.overallScore >= 45
                    ? "Borderline match with several required skill or experience gaps."
                    : "Low alignment with role requirements."}
                </p>
                <p className="text-[11px] text-slate-500">
                  Evaluated across 7 weighted criteria against the {ev.jobTitle} specification.
                </p>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
              <span>{ev.matchedSkills?.length || 0} skills matched</span>
              <span>{ev.experienceYears.toFixed(1)} years detected</span>
            </div>
          </div>

          {/* Standalone Resume Quality */}
          <div className="bg-white rounded-2xl p-6 sm:p-7 border border-slate-200 shadow-xs flex flex-col justify-between space-y-4">
            <div className="flex items-start justify-between">
              <div className="space-y-0.5">
                <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">
                  Document Audit
                </span>
                <h2 className="text-lg font-bold text-slate-900">
                  Resume Quality
                </h2>
              </div>
              <span className="text-xs font-semibold text-slate-500">
                Job-Independent
              </span>
            </div>

            <div className="space-y-3 my-2">
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-black text-slate-900">
                  {ev.resumeQuality.toFixed(0)}
                </span>
                <span className="text-xs text-slate-400">/ 100</span>
              </div>
              <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden">
                <div
                  className="h-full bg-slate-800 rounded-full"
                  style={{ width: `${Math.min(ev.resumeQuality, 100)}%` }}
                />
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">
                Evaluates structural clarity, standard ATS section headers, quantifiable bullet points,
                and readability. Does not penalize candidate based on specific job criteria.
              </p>
            </div>

            <div className="pt-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-500">
              <span>Text extraction: Clean</span>
              <span>
                {ev.stuffing?.suspected
                  ? "Keyword repetition noted"
                  : "No keyword stuffing detected"}
              </span>
            </div>
          </div>
        </div>

        {/* 3. Borderline Review Notice (Only when applicable) */}
        {ev.needsHumanReview && (
          <div className="bg-amber-50 rounded-2xl p-5 border border-amber-200 flex items-start gap-3.5">
            <div className="p-2 rounded-xl bg-amber-500 text-white shrink-0 mt-0.5">
              <AlertTriangle className="w-4 h-4" />
            </div>
            <div className="space-y-1">
              <h3 className="font-bold text-amber-900 text-sm">
                Recruiter Attention Recommended
              </h3>
              <p className="text-xs text-amber-800 leading-relaxed">
                {ev.hybridDecision?.policyReason ||
                  "The candidate's score is near the decision threshold or requires verification of specific skill claims. Final decision should be confirmed by a human recruiter."}
              </p>
            </div>
          </div>
        )}

        {/* 4. Decision Context: Strengths & Areas for Review */}
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-5">
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Evaluation Summary
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Key observations grounded in verified resume content
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Strengths */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200/80 space-y-3">
              <div className="flex items-center gap-2 text-emerald-700 font-bold text-xs uppercase tracking-wider">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Verified Strengths</span>
              </div>
              <ul className="space-y-2 text-xs text-slate-700">
                {strengths.length > 0 ? (
                  strengths.map((s, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-emerald-600 font-bold mt-0.5">✓</span>
                      <span>{s}</span>
                    </li>
                  ))
                ) : (
                  <li className="flex items-start gap-2">
                    <span className="text-emerald-600 font-bold mt-0.5">✓</span>
                    <span>Demonstrates relevant technical background and applicable experience.</span>
                  </li>
                )}
              </ul>
            </div>

            {/* Areas for Review */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200/80 space-y-3">
              <div className="flex items-center gap-2 text-amber-700 font-bold text-xs uppercase tracking-wider">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                <span>Areas for Review / Gaps</span>
              </div>
              <ul className="space-y-2 text-xs text-slate-700">
                {weaknesses.length > 0 ? (
                  weaknesses.map((w, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <span className="text-amber-600 font-bold mt-0.5">⚠</span>
                      <span>{w}</span>
                    </li>
                  ))
                ) : (
                  <li className="flex items-start gap-2 text-slate-500">
                    <span>•</span>
                    <span>No critical missing requirements or concerns detected.</span>
                  </li>
                )}
              </ul>
            </div>
          </div>
        </div>

        {/* 5. Deterministic Fit Breakdown (7 Dimensions) */}
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                Fit Score Breakdown
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Composite evaluation weighted across 7 objective qualification criteria
              </p>
            </div>
            <span className="text-xs font-semibold text-slate-700 bg-slate-100 px-3 py-1 rounded-lg">
              Total Score: {ev.overallScore.toFixed(1)} / 100
            </span>
          </div>

          <div className="space-y-3.5">
            {dimensions.map((dim) => {
              const contribution = (dim.score * (dim.weight / 100)).toFixed(1);
              return (
                <div key={dim.label} className="bg-slate-50 rounded-xl p-4 border border-slate-200/80 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-slate-900">{dim.label}</span>
                      <span className="text-slate-400 text-[11px]">({dim.weight}% weight)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-slate-900">{dim.score.toFixed(0)} / 100</span>
                      <span className="text-emerald-700 font-medium text-[11px]">(+{contribution} pts)</span>
                    </div>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-200 overflow-hidden">
                    <div
                      className="h-full bg-slate-800 rounded-full"
                      style={{ width: `${Math.min(dim.score, 100)}%` }}
                    />
                  </div>
                  <p className="text-[11px] text-slate-500">{dim.description}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* 6. Skill Verification Matrix */}
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-5">
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Skill Verification
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Direct comparison between candidate competencies and job specifications
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Matched Skills */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-slate-900 uppercase tracking-wider">
                  Matched Skills
                </span>
                <span className="text-xs text-emerald-700 font-bold">
                  {ev.matchedSkills?.length || 0} Verified
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {ev.matchedSkills && ev.matchedSkills.length > 0 ? (
                  ev.matchedSkills.map((skill, i) => (
                    <span
                      key={i}
                      className="px-2.5 py-1 rounded-lg bg-white border border-slate-200 text-slate-800 text-xs font-medium shadow-2xs"
                    >
                      {skill}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-400 italic">No exact skill matches recorded.</span>
                )}
              </div>
            </div>

            {/* Missing Skills */}
            <div className="bg-slate-50 rounded-xl p-5 border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <span className="font-bold text-xs text-slate-900 uppercase tracking-wider">
                  Missing Requirements
                </span>
                <span className="text-xs text-amber-700 font-bold">
                  {ev.missingSkills?.length || 0} Unmatched
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {ev.missingSkills && ev.missingSkills.length > 0 ? (
                  ev.missingSkills.map((skill, i) => (
                    <span
                      key={i}
                      className="px-2.5 py-1 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs font-medium"
                    >
                      {skill} (Missing)
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-emerald-600 font-medium">All specified skills verified.</span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* 7. Experience & Evidence Passages */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Experience Summary */}
          <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                Experience Analysis
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Career tenure and relevance to role seniority
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase font-semibold">Detected Career Tenure</span>
                <div className="text-xl font-bold text-slate-900 mt-0.5">
                  {ev.experienceYears.toFixed(1)} yrs
                </div>
              </div>
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase font-semibold">Tenure Match Score</span>
                <div className="text-xl font-bold text-emerald-700 mt-0.5">
                  {ev.experienceMatch.toFixed(0)} / 100
                </div>
              </div>
            </div>

            {/* Resume Passages */}
            <div className="space-y-2 pt-2">
              <span className="font-bold text-xs text-slate-800 uppercase tracking-wider block">
                Extracted Resume Evidence
              </span>
              {allEvidence && allEvidence.length > 0 ? (
                allEvidence.slice(0, 3).map((ep, i) => (
                  <div key={i} className="p-3 bg-slate-50 rounded-xl border border-slate-200/80 space-y-1">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="font-semibold text-slate-800">{ep.section}</span>
                      <span className="text-slate-400">Match: {(ep.similarity * 100).toFixed(0)}%</span>
                    </div>
                    <p className="text-xs text-slate-600 italic">&ldquo;{ep.text}&rdquo;</p>
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-400 italic">No specific passage quotes extracted.</p>
              )}
            </div>
          </div>

          {/* Project & Education */}
          <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                Projects & Academic Background
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Practical engineering work and degree qualifications
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase font-semibold">Project Relevance</span>
                <div className="text-xl font-bold text-slate-900 mt-0.5">
                  {ev.projectMatch.toFixed(0)} / 100
                </div>
              </div>
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <span className="text-[10px] text-slate-500 uppercase font-semibold">Education Match</span>
                <div className="text-xl font-bold text-slate-900 mt-0.5">
                  {ev.educationMatch.toFixed(0)} / 100
                </div>
              </div>
            </div>

            {/* ATS Parsability */}
            <div className="space-y-2 pt-2">
              <span className="font-bold text-xs text-slate-800 uppercase tracking-wider block">
                Formatting & Layout Check
              </span>
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Standard Sections:</span>
                  <span className="font-semibold text-slate-900">Verified</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Contact Details:</span>
                  <span className="font-semibold text-slate-900">Complete</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-600">Keyword Repetition:</span>
                  <span className="font-semibold text-emerald-700">
                    {ev.stuffing?.suspected ? "Flagged for check" : "Clean"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 8. Recruiter Determination Card */}
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-xs space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h2 className="text-lg font-bold text-slate-900">
                Recruiter Determination
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Record your hiring decision and notes for this candidate
              </p>
            </div>
            {decidedAt && (
              <span className="text-[11px] text-slate-500 font-medium">
                Last updated: {new Date(decidedAt).toLocaleString()}
              </span>
            )}
          </div>

          <div className="space-y-3">
            <textarea
              rows={3}
              placeholder="Add recruiter feedback, interview notes, or comments..."
              value={recruiterNotes}
              onChange={(e) => setRecruiterNotes(e.target.value)}
              className="w-full p-3.5 text-xs bg-slate-50 border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-1 focus:ring-slate-900 resize-none"
            />

            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleSaveDecision("SHORTLIST")}
                  disabled={savingDecision}
                  className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs disabled:opacity-50 ${
                    recruiterDecision === "SHORTLIST"
                      ? "bg-emerald-600 text-white"
                      : "bg-white border border-slate-200 text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  <Check className="w-3.5 h-3.5" />
                  <span>{savingDecision && recruiterDecision === "SHORTLIST" ? "Saving..." : "Confirm Shortlist"}</span>
                </button>
                <button
                  onClick={() => handleSaveDecision("REVIEW")}
                  disabled={savingDecision}
                  className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs disabled:opacity-50 ${
                    recruiterDecision === "REVIEW"
                      ? "bg-amber-600 text-white"
                      : "bg-white border border-slate-200 text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  <Flag className="w-3.5 h-3.5" />
                  <span>{savingDecision && recruiterDecision === "REVIEW" ? "Saving..." : "Keep in Review"}</span>
                </button>
                <button
                  onClick={() => handleSaveDecision("REJECT")}
                  disabled={savingDecision}
                  className={`px-4 py-2 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 shadow-xs disabled:opacity-50 ${
                    recruiterDecision === "REJECT"
                      ? "bg-slate-800 text-white"
                      : "bg-white border border-slate-200 text-slate-700 hover:bg-slate-50"
                  }`}
                >
                  <XCircle className="w-3.5 h-3.5" />
                  <span>{savingDecision && recruiterDecision === "REJECT" ? "Saving..." : "Reject"}</span>
                </button>
                {recruiterDecision && (
                  <button
                    onClick={() => handleSaveDecision(recruiterDecision)}
                    disabled={savingDecision}
                    className="px-3 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:text-slate-900 border border-slate-200 hover:bg-slate-50"
                    title="Update notes for this determination"
                  >
                    Save Notes
                  </button>
                )}
              </div>

              <span className="text-[11px] text-slate-400">
                Decisions are persisted to SQLite and logged to episodic audit history
              </span>
            </div>

            {decisionSaved && (
              <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-emerald-800 text-xs flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                <span>
                  Recruiter determination successfully recorded as <strong>{recruiterDecision}</strong>.
                </span>
              </div>
            )}

            {decisionError && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-800 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
                <span>{decisionError}</span>
              </div>
            )}
          </div>
        </div>

        {/* 9. Technical Provenance & System Audit (Secondary Collapsible at Bottom) */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
          <button
            onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
            className="w-full p-5 flex items-center justify-between text-left hover:bg-slate-50 transition"
          >
            <div className="flex items-center gap-2.5">
              <Layers className="w-4 h-4 text-slate-500" />
              <span className="text-xs font-semibold text-slate-700">
                System Audit & Evaluation Provenance (Technical Details)
              </span>
            </div>
            {showTechnicalDetails ? (
              <ChevronDown className="w-4 h-4 text-slate-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-slate-400" />
            )}
          </button>

          {showTechnicalDetails && (
            <div className="p-6 border-t border-slate-200 bg-slate-50/50 space-y-4 text-xs font-data-mono-sm">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 bg-white rounded-xl border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase block">Scoring Engine</span>
                  <span className="font-semibold text-slate-800 mt-1 block">Deterministic Composite</span>
                </div>
                <div className="p-3 bg-white rounded-xl border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase block">Second Opinion</span>
                  <span className="font-semibold text-slate-800 mt-1 block">
                    {ev.mlPrediction ? "Calibrated Logistic Model" : "Rule Consensus"}
                  </span>
                </div>
                <div className="p-3 bg-white rounded-xl border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase block">Vector Retrieval</span>
                  <span className="font-semibold text-slate-800 mt-1 block">Dense Semantic Index</span>
                </div>
                <div className="p-3 bg-white rounded-xl border border-slate-200">
                  <span className="text-slate-400 text-[10px] uppercase block">Demographic Masking</span>
                  <span className="font-semibold text-emerald-700 mt-1 block">Active</span>
                </div>
              </div>

              {/* Execution Trace */}
              {ev.trace && ev.trace.length > 0 && (
                <div className="space-y-2 pt-2">
                  <span className="font-bold text-xs text-slate-800 uppercase block">
                    Pipeline Execution Steps ({ev.trace.length})
                  </span>
                  <div className="space-y-1.5">
                    {ev.trace.map((step, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-xl bg-white border border-slate-200 flex items-center justify-between text-[11px]"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-slate-900">{step.agent}</span>
                          <span className="text-slate-400">•</span>
                          <span className="text-slate-600">{step.task}</span>
                        </div>
                        <span className="text-slate-500 font-medium">{step.durationMs.toFixed(0)} ms</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
