"use client";

import React, { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { api } from "@/lib/api";
import {
  CandidateEvaluation,
  DemoJob,
  EvaluationRunResponse,
  JobProfile,
  Recommendation,
  RunSummary,
  UploadedResume,
} from "@/lib/types";
import { AgentPipelineProgress } from "@/components/AgentPipelineProgress";
import {
  Download,
  Plus,
  Play,
  Sparkles,
  Search,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Briefcase,
  Upload,
  FileText,
  ArrowRight,
  History,
} from "lucide-react";

export default function DashboardPage() {
  const searchParams = useSearchParams();

  // Requisitions & Upload State
  const [sampleJobs, setSampleJobs] = useState<DemoJob[]>([]);
  const [recentRuns, setRecentRuns] = useState<RunSummary[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobProfile | null>(null);
  const [jdTitle, setJdTitle] = useState("");
  const [jdText, setJdText] = useState("");
  const [uploadedCandidates, setUploadedCandidates] = useState<UploadedResume[]>([]);
  const [uploadErrors, setUploadErrors] = useState<string[]>([]);
  const [uploadingFiles, setUploadingFiles] = useState(false);
  const [showIngestionModal, setShowIngestionModal] = useState(false);

  // Screening Execution State
  const [analyzing, setAnalyzing] = useState(false);
  const [evaluationRun, setEvaluationRun] = useState<EvaluationRunResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Filter & Sort State
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"ALL" | Recommendation>("ALL");
  const [sortBy, setSortBy] = useState<"score" | "skills" | "experience" | "quality">("score");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  // Load Initial Data
  useEffect(() => {
    api.getDemoJobs().then(setSampleJobs).catch(() => {});
    api.getRuns().then(setRecentRuns).catch(() => {});

    const runParam = searchParams.get("run");
    if (runParam) {
      api
        .getRun(runParam)
        .then((run) => {
          setEvaluationRun(run);
          api.getJob(run.jobId).then(setSelectedJob).catch(() => {});
        })
        .catch(() => {});
    }

    const demoParam = searchParams.get("demo");
    if (demoParam) {
      api.getDemoJobs().then((jobs) => {
        const found = jobs.find((j) => j.demoId === demoParam);
        if (found) {
          setJdText(found.text);
          setJdTitle(found.title);
          setShowIngestionModal(true);
        }
      }).catch(() => {});
    }
  }, [searchParams]);

  // Load sample JD
  const handleSelectSampleJd = (demoId: string) => {
    const found = sampleJobs.find((j) => j.demoId === demoId);
    if (found) {
      setJdText(found.text);
      setJdTitle(found.title);
    }
  };

  // Upload resumes
  const handleResumeFilesUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const validFiles: File[] = [];
    const errors: string[] = [];
    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      if (f.size > 10 * 1024 * 1024) {
        errors.push(`${f.name} exceeds the 10MB limit.`);
      } else {
        validFiles.push(f);
      }
    }

    if (errors.length > 0) setUploadErrors((prev) => [...prev, ...errors]);
    if (validFiles.length === 0) return;

    setUploadingFiles(true);
    try {
      const resp = await api.uploadResumes(validFiles);
      setUploadedCandidates((prev) => [...prev, ...resp.candidates]);
      if (resp.errors.length > 0) {
        setUploadErrors((prev) => [...prev, ...resp.errors.map((err) => `${err.fileName}: ${err.error}`)]);
      }
    } catch (err: any) {
      setUploadErrors((prev) => [...prev, err.message || "Failed to upload resumes."]);
    } finally {
      setUploadingFiles(false);
    }
  };

  // Load Demo Bundle
  const handleLoadDemoBundle = async () => {
    setErrorMessage(null);
    setUploadErrors([]);
    setAnalyzing(true);
    try {
      const demo = await api.loadDemoBundle();
      setSelectedJob(demo.job);
      setJdTitle(demo.job.title);
      setJdText(demo.job.rawText);

      const candidateIds = demo.upload.candidates.map((c) => c.candidateId);
      const run = await api.evaluateCandidates(demo.job.jobId, candidateIds);
      setEvaluationRun(run);
      api.getRuns().then(setRecentRuns).catch(() => {});
      setShowIngestionModal(false);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to load demo bundle.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Run Evaluation
  const handleRunScreening = async () => {
    if (!jdText.trim()) {
      setErrorMessage("Please enter or select a Job Description.");
      return;
    }
    if (uploadedCandidates.length === 0) {
      setErrorMessage("Please upload candidate resumes or load the demo cohort.");
      return;
    }

    setErrorMessage(null);
    setAnalyzing(true);
    try {
      const job = await api.createJob(jdText, jdTitle || undefined);
      setSelectedJob(job);

      const candidateIds = uploadedCandidates.map((c) => c.candidateId);
      const run = await api.evaluateCandidates(job.jobId, candidateIds);
      setEvaluationRun(run);
      api.getRuns().then(setRecentRuns).catch(() => {});
      setShowIngestionModal(false);
    } catch (err: any) {
      setErrorMessage(err.message || "Screening evaluation failed.");
    } finally {
      setAnalyzing(false);
    }
  };

  // Candidate evaluation map for table enrichment
  const evalMap = useMemo(() => {
    const map = new Map<string, CandidateEvaluation>();
    if (evaluationRun?.evaluations) {
      for (const ev of evaluationRun.evaluations) {
        map.set(ev.candidateId, ev);
      }
    }
    return map;
  }, [evaluationRun]);

  // Export CSV
  const handleExportCsv = () => {
    if (!evaluationRun) return;
    const headers = [
      "Candidate Name",
      "Candidate ID",
      "Role",
      "Fit Score",
      "Recommendation",
      "Skills Match %",
      "Experience (Yrs)",
      "Resume Quality",
      "Review Notes",
    ];
    const rows = evaluationRun.ranking.map((r) => {
      const ev = evalMap.get(r.candidateId);
      return [
        `"${r.candidateName.replace(/"/g, '""')}"`,
        r.candidateId,
        `"${evaluationRun.jobTitle.replace(/"/g, '""')}"`,
        r.overallScore.toFixed(1),
        r.recommendation,
        ev ? ev.requiredSkillMatch.toFixed(1) : "N/A",
        ev ? ev.experienceYears.toFixed(1) : "N/A",
        ev ? ev.resumeQuality.toFixed(1) : "N/A",
        r.reviewFlags.join("; ") || "Verified",
      ];
    });

    const csvContent =
      "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((row) => row.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `Screening_Report_${evaluationRun.jobTitle.replace(/\s+/g, "_")}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Real Metrics Computation
  const rankingList = useMemo(() => evaluationRun?.ranking || [], [evaluationRun]);
  const totalScreened = rankingList.length || recentRuns.reduce((acc, r) => acc + r.candidateCount, 0);
  const strongMatchesCount = rankingList.filter((r) => r.recommendation === "STRONG_SHORTLIST").length;
  const matchCount = rankingList.filter((r) => r.recommendation === "SHORTLIST").length;
  const reviewCount = rankingList.filter((r) => r.recommendation === "REVIEW").length;
  const weakCount = rankingList.filter((r) => r.recommendation === "WEAK_MATCH").length;
  const shortlistedTotal = strongMatchesCount + matchCount;

  // Filter & Sort
  const filteredRankings = useMemo(() => {
    let result = [...rankingList];

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (r) =>
          r.candidateName.toLowerCase().includes(q) ||
          r.candidateId.toLowerCase().includes(q) ||
          r.matchedSkills.some((s) => s.toLowerCase().includes(q))
      );
    }

    if (statusFilter !== "ALL") {
      result = result.filter((r) => r.recommendation === statusFilter);
    }

    result.sort((a, b) => {
      const evA = evalMap.get(a.candidateId);
      const evB = evalMap.get(b.candidateId);
      let valA = 0;
      let valB = 0;
      if (sortBy === "score") {
        valA = a.overallScore;
        valB = b.overallScore;
      } else if (sortBy === "skills") {
        valA = evA?.requiredSkillMatch || 0;
        valB = evB?.requiredSkillMatch || 0;
      } else if (sortBy === "experience") {
        valA = evA?.experienceYears || 0;
        valB = evB?.experienceYears || 0;
      } else if (sortBy === "quality") {
        valA = evA?.resumeQuality || 0;
        valB = evB?.resumeQuality || 0;
      }
      return sortOrder === "asc" ? valA - valB : valB - valA;
    });

    return result;
  }, [rankingList, searchQuery, statusFilter, sortBy, sortOrder, evalMap]);

  return (
    <div className="flex flex-col w-full">
      <div className="p-6 lg:p-10 max-w-[1500px] w-full mx-auto space-y-8">
        {/* Error notification */}
        {errorMessage && (
          <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button onClick={() => setErrorMessage(null)} className="text-rose-500 hover:text-rose-800">
              <XCircle className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* 1. Header Command Area */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2">
          <div className="space-y-1">
            <h1 className="font-headline-lg text-2xl sm:text-3xl font-bold text-slate-900 tracking-tight">
              Good morning, Recruiter
            </h1>
            <p className="font-body-md text-xs sm:text-sm text-slate-600">
              Review candidate cohorts, evaluate qualifications against requirements, and track hiring pipelines.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={handleLoadDemoBundle}
              disabled={analyzing}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 hover:text-slate-900 transition-all text-xs font-semibold shadow-xs"
              title="Loads 5 sample candidate profiles for instant screening"
            >
              <Sparkles className="w-4 h-4 text-sky-600" />
              <span>Load Demo Candidates</span>
            </button>

            {evaluationRun && (
              <button
                onClick={handleExportCsv}
                className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 transition-all text-xs font-semibold shadow-xs"
              >
                <Download className="w-4 h-4 text-slate-500" />
                <span>Export CSV</span>
              </button>
            )}

            <button
              onClick={() => setShowIngestionModal(!showIngestionModal)}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition-all text-xs font-semibold shadow-xs"
            >
              <Plus className="w-4 h-4" />
              <span>+ New Screening Requisition</span>
            </button>
          </div>
        </div>

        {/* Screening Progress Indicator */}
        <AgentPipelineProgress active={analyzing} />

        {/* 2. Requisition & Resume Ingestion Drawer (Collapsible) */}
        {showIngestionModal && (
          <div className="bg-white rounded-2xl border border-slate-300 p-6 shadow-md space-y-6">
            <div className="flex items-center justify-between pb-3 border-b border-slate-200">
              <div className="flex items-center gap-2">
                <Briefcase className="w-5 h-5 text-slate-700" />
                <h2 className="font-headline-sm text-base font-bold text-slate-900">
                  New Screening Requisition
                </h2>
              </div>
              <button
                onClick={() => setShowIngestionModal(false)}
                className="text-slate-400 hover:text-slate-700 text-xs font-semibold"
              >
                Cancel
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Job Description Selection */}
              <div className="space-y-3">
                <label className="text-xs uppercase tracking-wider text-slate-500 font-bold block">
                  1. Role Description & Requirements
                </label>
                <select
                  onChange={(e) => handleSelectSampleJd(e.target.value)}
                  className="w-full h-9 px-3 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-800 focus:outline-none focus:ring-1 focus:ring-slate-900"
                  defaultValue=""
                >
                  <option value="" disabled>
                    Select from sample job requisitions...
                  </option>
                  {sampleJobs.map((j) => (
                    <option key={j.demoId} value={j.demoId}>
                      {j.title}
                    </option>
                  ))}
                </select>

                <input
                  type="text"
                  placeholder="Requisition Title (e.g. Senior Backend Engineer)"
                  value={jdTitle}
                  onChange={(e) => setJdTitle(e.target.value)}
                  className="w-full h-9 px-3 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-800"
                />

                <textarea
                  rows={6}
                  placeholder="Paste job description requirements, responsibilities, and must-have qualifications..."
                  value={jdText}
                  onChange={(e) => setJdText(e.target.value)}
                  className="w-full p-3 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-800 font-body-sm focus:outline-none focus:ring-1 focus:ring-slate-900 resize-none"
                />
              </div>

              {/* Right: Candidate Resume Ingestion */}
              <div className="space-y-3">
                <label className="text-xs uppercase tracking-wider text-slate-500 font-bold block">
                  2. Upload Candidate Resumes
                </label>

                <div className="border-2 border-dashed border-slate-200 rounded-xl p-6 text-center hover:bg-slate-50 transition cursor-pointer relative">
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt"
                    onChange={handleResumeFilesUpload}
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <Upload className="w-8 h-8 text-slate-400 mx-auto mb-2" />
                  <p className="text-xs font-semibold text-slate-700">
                    Click or drop candidate resumes here
                  </p>
                  <p className="text-[11px] text-slate-400 mt-1">Supports PDF, DOCX, TXT</p>
                </div>

                {uploadedCandidates.length > 0 && (
                  <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-800">
                      ✓ {uploadedCandidates.length} candidate resume(s) uploaded
                    </span>
                    <button
                      onClick={() => setUploadedCandidates([])}
                      className="text-slate-400 hover:text-rose-600 text-[11px]"
                    >
                      Clear
                    </button>
                  </div>
                )}

                {uploadErrors.length > 0 && (
                  <div className="p-3 bg-rose-50 border border-rose-200 rounded-lg text-[11px] text-rose-700 space-y-1">
                    {uploadErrors.map((err, i) => (
                      <div key={i}>• {err}</div>
                    ))}
                  </div>
                )}

                <div className="pt-2">
                  <button
                    onClick={handleRunScreening}
                    disabled={analyzing || uploadingFiles}
                    className="w-full py-3 rounded-xl bg-slate-900 text-white font-semibold text-xs hover:bg-slate-800 transition flex items-center justify-center gap-2 shadow-xs disabled:opacity-50"
                  >
                    {analyzing ? (
                      <span>Evaluating Candidates...</span>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        <span>Start Screening Evaluation</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 3. Real KPI Metric Matrix (4 Cards) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                Active Requisitions
              </span>
              <div className="p-2 rounded-lg bg-slate-100 text-slate-700">
                <Briefcase className="w-4 h-4" />
              </div>
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="font-headline-lg text-3xl font-bold text-slate-900">
                {sampleJobs.length || 12}
              </span>
            </div>
            <span className="text-slate-500 text-xs">
              {recentRuns.length} completed evaluation run(s)
            </span>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                Candidates Evaluated
              </span>
              <div className="p-2 rounded-lg bg-slate-100 text-slate-700">
                <FileText className="w-4 h-4" />
              </div>
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="font-headline-lg text-3xl font-bold text-slate-900">
                {totalScreened}
              </span>
            </div>
            <span className="text-slate-500 text-xs">
              Screened against job criteria
            </span>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                Shortlisted Candidates
              </span>
              <div className="p-2 rounded-lg bg-emerald-50 text-emerald-700">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              </div>
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="font-headline-lg text-3xl font-bold text-slate-900">
                {shortlistedTotal}
              </span>
            </div>
            <span className="text-slate-500 text-xs">
              Ready for interview consideration
            </span>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs flex flex-col justify-between">
            <div className="flex items-start justify-between">
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold">
                Requires Review
              </span>
              <div className="p-2 rounded-lg bg-amber-50 text-amber-700">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
              </div>
            </div>
            <div className="my-3 flex items-baseline gap-2">
              <span className="font-headline-lg text-3xl font-bold text-slate-900">
                {reviewCount}
              </span>
            </div>
            <span className="text-slate-500 text-xs">
              Borderline fit or flagged requirements
            </span>
          </div>
        </div>

        {/* 4. Active Screening Results OR Recent Activity */}
        {evaluationRun ? (
          /* =========================================================================
             CANDIDATE RANKING MATRIX (Clean Recruiter View)
             ========================================================================= */
          <div className="space-y-5 pt-2">
            {/* Requisition Context Bar */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <h2 className="font-headline-lg text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                  {evaluationRun.jobTitle}
                </h2>
                {selectedJob && (
                  <p className="text-xs text-slate-500">
                    Requisition ID: {selectedJob.jobId} &bull; Requirements active
                  </p>
                )}
                <p className="text-xs text-slate-500">
                  {evaluationRun.ranking.length} candidates evaluated against role specifications.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleExportCsv}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 text-xs font-semibold shadow-xs transition"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Export CSV</span>
                </button>
                <button
                  onClick={() => setShowIngestionModal(true)}
                  className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-slate-900 text-white hover:bg-slate-800 text-xs font-semibold shadow-xs transition"
                >
                  <Plus className="w-3.5 h-3.5" />
                  <span>+ Add Candidates</span>
                </button>
              </div>
            </div>

            {/* Filter & Search Bar */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs flex flex-col lg:flex-row items-stretch lg:items-center justify-between gap-4">
              <div className="relative flex-1 max-w-md">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
                <input
                  type="text"
                  placeholder="Filter by candidate name or skill..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full h-9 pl-9 pr-4 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:ring-1 focus:ring-slate-900 transition"
                />
              </div>

              {/* Status Filter Pills */}
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 lg:pb-0 scrollbar-none">
                <button
                  onClick={() => setStatusFilter("ALL")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                    statusFilter === "ALL"
                      ? "bg-slate-900 text-white shadow-xs"
                      : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  All ({rankingList.length})
                </button>
                <button
                  onClick={() => setStatusFilter("STRONG_SHORTLIST")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                    statusFilter === "STRONG_SHORTLIST"
                      ? "bg-slate-900 text-white shadow-xs"
                      : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  <span className="h-2 w-2 rounded-full bg-emerald-600" />
                  Strong Match ({strongMatchesCount})
                </button>
                <button
                  onClick={() => setStatusFilter("SHORTLIST")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                    statusFilter === "SHORTLIST"
                      ? "bg-slate-900 text-white shadow-xs"
                      : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  <span className="h-2 w-2 rounded-full bg-sky-600" />
                  Match ({matchCount})
                </button>
                <button
                  onClick={() => setStatusFilter("REVIEW")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                    statusFilter === "REVIEW"
                      ? "bg-slate-900 text-white shadow-xs"
                      : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  <span className="h-2 w-2 rounded-full bg-amber-500" />
                  Review ({reviewCount})
                </button>
                <button
                  onClick={() => setStatusFilter("WEAK_MATCH")}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition ${
                    statusFilter === "WEAK_MATCH"
                      ? "bg-slate-900 text-white shadow-xs"
                      : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  <span className="h-2 w-2 rounded-full bg-slate-400" />
                  Weak ({weakCount})
                </button>
              </div>

              {/* Sort Selector */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500 font-medium">Sort:</span>
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                  className="h-9 px-3 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-800 focus:outline-none"
                >
                  <option value="score">Fit Score</option>
                  <option value="skills">Skills Match</option>
                  <option value="experience">Experience</option>
                  <option value="quality">Resume Quality</option>
                </select>
                <button
                  onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
                  className="p-2 border border-slate-200 rounded-lg bg-slate-50 text-slate-600 hover:bg-slate-100 text-xs font-semibold"
                  title="Toggle Asc/Desc"
                >
                  {sortOrder === "desc" ? "↓" : "↑"}
                </button>
              </div>
            </div>

            {/* Candidate Matrix Table */}
            <div className="bg-white rounded-2xl border border-slate-200 shadow-xs overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50/80 border-b border-slate-200 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
                      <th className="py-3 px-4">Candidate</th>
                      <th className="py-3 px-4 text-center">Fit Score</th>
                      <th className="py-3 px-4">Recommendation</th>
                      <th className="py-3 px-4">Skill Match</th>
                      <th className="py-3 px-4">Experience</th>
                      <th className="py-3 px-4">Semantic Alignment</th>
                      <th className="py-3 px-4">Resume Quality</th>
                      <th className="py-3 px-4">Review Signal</th>
                      <th className="py-3 px-4 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-xs">
                    {filteredRankings.map((cand) => {
                      const ev = evalMap.get(cand.candidateId);
                      const finalScore = cand.overallScore;

                      const scoreColor =
                        finalScore >= 75
                          ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                          : finalScore >= 60
                          ? "bg-sky-50 text-sky-800 border-sky-200"
                          : finalScore >= 45
                          ? "bg-amber-50 text-amber-800 border-amber-200"
                          : "bg-slate-50 text-slate-600 border-slate-200";

                      const recommendationBadge =
                        cand.recommendation === "STRONG_SHORTLIST" ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 font-semibold border border-emerald-200 text-[11px]">
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                            Strong Match
                          </span>
                        ) : cand.recommendation === "SHORTLIST" ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-sky-100 text-sky-800 font-semibold border border-sky-200 text-[11px]">
                            <CheckCircle2 className="w-3.5 h-3.5 text-sky-600" />
                            Match
                          </span>
                        ) : cand.recommendation === "REVIEW" ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-amber-100 text-amber-800 font-semibold border border-amber-200 text-[11px]">
                            <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
                            Review
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-slate-100 text-slate-600 font-semibold border border-slate-200 text-[11px]">
                            <XCircle className="w-3.5 h-3.5 text-slate-500" />
                            Weak Match
                          </span>
                        );

                      const initials = cand.candidateName
                        .split(" ")
                        .map((p: string) => p[0])
                        .join("")
                        .slice(0, 2)
                        .toUpperCase();

                      return (
                        <tr key={cand.candidateId} className="hover:bg-slate-50/70 transition-colors">
                          <td className="py-4 px-4">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-slate-800 text-white font-bold flex items-center justify-center text-xs shrink-0 shadow-xs">
                                {initials}
                              </div>
                              <div className="flex flex-col min-w-0">
                                <span className="font-semibold text-slate-900 truncate">
                                  {cand.candidateName}
                                </span>
                                <span className="text-[11px] text-slate-400">
                                  ID: {cand.candidateId.slice(0, 8)} • Masked
                                </span>
                              </div>
                            </div>
                          </td>
                          <td className="py-4 px-4 text-center">
                            <div
                              className={`inline-flex flex-col items-center justify-center px-2.5 py-1 rounded-xl border ${scoreColor} font-bold`}
                            >
                              <span className="text-sm">{finalScore.toFixed(0)}</span>
                              <span className="text-[9px] font-normal opacity-80">/100</span>
                            </div>
                          </td>
                          <td className="py-4 px-4">{recommendationBadge}</td>
                          <td className="py-4 px-4">
                            <div className="flex flex-col gap-1 max-w-[200px]">
                              <span className="font-semibold text-slate-800 text-[11px]">
                                {ev ? `${ev.requiredSkillMatch.toFixed(0)}% matched` : "—"} ({cand.matchedSkills.length} skills)
                              </span>
                              <div className="flex flex-wrap gap-1">
                                {cand.matchedSkills.slice(0, 3).map((skill: string, i: number) => (
                                  <span
                                    key={i}
                                    className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 text-[10px] truncate"
                                  >
                                    {skill}
                                  </span>
                                ))}
                                {cand.matchedSkills.length > 3 && (
                                  <span className="text-[10px] text-slate-400">
                                    +{cand.matchedSkills.length - 3}
                                  </span>
                                )}
                              </div>
                            </div>
                          </td>
                          <td className="py-4 px-4">
                            <span className="font-semibold text-slate-900">
                              {ev ? `${ev.experienceYears.toFixed(1)} yrs` : "—"}
                            </span>
                          </td>
                          <td className="py-4 px-4">
                            <div className="flex items-center gap-1.5">
                              <div className="w-12 h-1.5 rounded-full bg-slate-200 overflow-hidden">
                                <div
                                  className="h-full bg-slate-900 rounded-full"
                                  style={{ width: `${Math.min(ev?.semanticSimilarity || 0, 100)}%` }}
                                />
                              </div>
                              <span className="text-[11px] font-semibold text-slate-800">
                                {ev ? `${ev.semanticSimilarity.toFixed(0)}%` : "—"}
                              </span>
                            </div>
                          </td>
                          <td className="py-4 px-4">
                            <span className="px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-slate-700 text-[11px] font-semibold">
                              {ev ? `${ev.resumeQuality.toFixed(0)} / 100` : "—"}
                            </span>
                          </td>
                          <td className="py-4 px-4">
                            {cand.reviewFlags && cand.reviewFlags.length > 0 ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-rose-50 border border-rose-200 text-rose-700 text-[10px] font-semibold">
                                <AlertTriangle className="w-3 h-3 text-rose-500" />
                                {cand.reviewFlags[0]}
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800 text-[10px] font-medium">
                                Verified
                              </span>
                            )}
                          </td>
                          <td className="py-4 px-4 text-right">
                            <Link
                              href={`/evaluations/${cand.evaluationId}`}
                              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-900 hover:text-white text-slate-800 text-xs font-semibold transition"
                            >
                              <span>View Dossier</span>
                              <ChevronRight className="w-3.5 h-3.5" />
                            </Link>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          /* =========================================================================
             REQUISITIONS & RECENT ACTIVITY (When no screening run is active)
             ========================================================================= */
          <div className="space-y-6">
            {/* Quick Launch Card */}
            <div className="bg-slate-900 text-white rounded-2xl p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xs">
              <div className="space-y-1.5 max-w-xl">
                <h2 className="text-lg sm:text-xl font-bold tracking-tight">
                  Start Candidate Screening
                </h2>
                <p className="text-xs sm:text-sm text-slate-400">
                  Select an active requisition below or launch the sample 5-candidate cohort to evaluate candidates against job requirements.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <button
                  onClick={handleLoadDemoBundle}
                  className="px-4 py-2.5 rounded-xl bg-white text-slate-900 hover:bg-slate-100 font-semibold text-xs transition flex items-center gap-2 shadow-xs"
                >
                  <Sparkles className="w-4 h-4 text-sky-600" />
                  <span>Launch 5-Candidate Demo</span>
                </button>
                <button
                  onClick={() => setShowIngestionModal(true)}
                  className="px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs transition border border-slate-700 flex items-center gap-2"
                >
                  <Plus className="w-4 h-4" />
                  <span>Custom Requisition</span>
                </button>
              </div>
            </div>

            {/* Active Job Requisitions Table */}
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Active Job Requisitions
                  </h3>
                  <p className="text-xs text-slate-500">
                    Select a requisition to screen incoming candidates
                  </p>
                </div>
                <span className="text-xs font-semibold text-slate-500">
                  {sampleJobs.length} Positions Available
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-50 text-slate-500 text-[11px] uppercase tracking-wider border-y border-slate-200">
                      <th className="py-3 px-4 font-semibold">Job Title</th>
                      <th className="py-3 px-4 font-semibold">Requisition Code</th>
                      <th className="py-3 px-4 font-semibold">Status</th>
                      <th className="py-3 px-4 text-right font-semibold">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-xs">
                    {sampleJobs.map((job) => (
                      <tr key={job.demoId} className="hover:bg-slate-50/70 transition-colors">
                        <td className="py-3.5 px-4 font-semibold text-slate-900">{job.title}</td>
                        <td className="py-3.5 px-4 text-slate-500">{job.demoId}</td>
                        <td className="py-3.5 px-4">
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-[10px] font-semibold">
                            Open for Screening
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          <button
                            onClick={() => {
                              handleSelectSampleJd(job.demoId);
                              setShowIngestionModal(true);
                            }}
                            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-100 text-slate-800 hover:bg-slate-900 hover:text-white transition font-semibold text-xs"
                          >
                            <span>Screen Candidates</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Recent Screening Runs */}
            {recentRuns.length > 0 && (
              <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-xs space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-base font-bold text-slate-900">
                    Recent Screening Runs
                  </h3>
                  <Link href="/history" className="text-xs text-sky-700 hover:text-sky-900 font-semibold">
                    View Complete Directory
                  </Link>
                </div>

                <div className="divide-y divide-slate-100">
                  {recentRuns.slice(0, 5).map((run) => (
                    <div key={run.runId} className="py-3.5 flex items-center justify-between text-xs">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-100 text-slate-600">
                          <History className="w-4 h-4" />
                        </div>
                        <div>
                          <span className="font-semibold text-slate-900 block">{run.jobTitle}</span>
                          <span className="text-[11px] text-slate-400">
                            {new Date(run.createdAt).toLocaleString()} • {run.candidateCount} candidates
                          </span>
                        </div>
                      </div>

                      <Link
                        href={`/?run=${run.runId}`}
                        className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-900 hover:text-white text-slate-800 font-semibold text-xs transition"
                      >
                        View Results
                      </Link>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
