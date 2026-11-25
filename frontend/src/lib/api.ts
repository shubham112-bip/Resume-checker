/**
 * Typed API Client for the FastAPI backend.
 */

import {
  CandidateDetail,
  CandidateEvaluation,
  CandidateSummary,
  ComparisonResponse,
  DemoJob,
  DemoLoadResponse,
  EpisodeResponse,
  EvaluationRunResponse,
  GraphResponse,
  HealthResponse,
  JobProfile,
  ModelMetricsResponse,
  ResumeQualityUploadResponse,
  RunSummary,
  UploadResponse,
  DatasetPairSummary,
  DatasetSummaryResponse,
  HumanValidationResponse,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  try {
    const res = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options?.headers || {}),
      },
      cache: "no-store",
    });

    if (!res.ok) {
      let errorMsg = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const body = await res.json();
        if (typeof body.detail === "string") {
          errorMsg = body.detail;
        } else if (Array.isArray(body.detail)) {
          errorMsg = body.detail.map((d: any) => d.msg || JSON.stringify(d)).join(", ");
        }
      } catch {
        // use default errorMsg
      }
      throw new ApiError(res.status, errorMsg);
    }

    return (await res.json()) as T;
  } catch (err: any) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      0,
      `Backend service is not reachable at ${API_BASE_URL}. Ensure the FastAPI server is running.`
    );
  }
}

export const api = {
  getHealth: () => request<HealthResponse>("/api/health"),

  getConfig: () => request<Record<string, any>>("/api/config"),

  getDemoJobs: () => request<DemoJob[]>("/api/demo/job-descriptions"),

  getJob: (jobId: string) => request<JobProfile>(`/api/jobs/${jobId}`),

  loadDemoBundle: () => request<DemoLoadResponse>("/api/demo/load", { method: "POST" }),

  createJob: (text: string, title?: string) =>
    request<JobProfile>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, title }),
    }),

  uploadJobFile: async (file: File): Promise<JobProfile> => {
    const formData = new FormData();
    formData.append("file", file);
    return request<JobProfile>("/api/jobs/upload", {
      method: "POST",
      body: formData,
    });
  },

  uploadResumes: async (files: File[]): Promise<UploadResponse> => {
    const formData = new FormData();
    for (const f of files) {
      formData.append("files", f);
    }
    return request<UploadResponse>("/api/resumes/upload", {
      method: "POST",
      body: formData,
    });
  },

  evaluateCandidates: (jobId: string, candidateIds: string[]) =>
    request<EvaluationRunResponse>("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ jobId, candidateIds }),
    }),

  getRuns: () => request<RunSummary[]>("/api/runs"),

  getRun: (runId: string) => request<EvaluationRunResponse>(`/api/runs/${runId}`),

  getCandidates: () => request<CandidateSummary[]>("/api/candidates"),

  getCandidate: (candidateId: string) =>
    request<CandidateDetail>(`/api/candidates/${candidateId}`),

  getEvaluation: (evaluationId: string) =>
    request<CandidateEvaluation>(`/api/evaluations/${evaluationId}`),

  getModelMetrics: (dataset: "synthetic" | "real" = "real") =>
    request<ModelMetricsResponse>(`/api/models/metrics?dataset=${dataset}`),

  runModelBenchmark: () =>
    request<ModelMetricsResponse>("/api/models/benchmark/run", { method: "POST" }),

  getDatasetSummary: () => request<DatasetSummaryResponse>("/api/dataset/summary"),

  getDatasetResumes: () => request<any[]>("/api/dataset/resumes"),

  getDatasetPairs: (limit = 100, resumeId?: string, jdId?: string) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (resumeId) params.append("resume_id", resumeId);
    if (jdId) params.append("jd_id", jdId);
    return request<DatasetPairSummary[]>(`/api/dataset/pairs?${params.toString()}`);
  },

  evaluateResumeQuality: async (file: File): Promise<ResumeQualityUploadResponse> => {
    const formData = new FormData();
    formData.append("file", file);
    return request<ResumeQualityUploadResponse>("/api/resumes/quality", {
      method: "POST",
      body: formData,
    });
  },

  getComparison: () => request<ComparisonResponse>("/api/comparison/ats-vs-ai"),

  getHumanValidation: () => request<HumanValidationResponse>("/api/evaluation/human-validation"),

  getEpisodes: () => request<EpisodeResponse[]>("/api/memory/episodes"),

  getAgentGraph: () => request<GraphResponse>("/api/agents/graph"),

  saveRecruiterDecision: (
    evaluationId: string,
    decision: "SHORTLIST" | "REVIEW" | "REJECT",
    notes: string = ""
  ) =>
    request<CandidateEvaluation>(`/api/evaluations/${evaluationId}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, notes }),
    }),

  getReportUrl: (evaluationId: string, format: "pdf" | "md" = "pdf") =>
    `${API_BASE_URL}/api/evaluations/${evaluationId}/report?format=${format}`,
};
