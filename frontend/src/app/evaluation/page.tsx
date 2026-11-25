"use client";

import React, { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { ComparisonResponse, HumanValidationResponse, ModelMetricsResponse } from "@/lib/types";
import { Card } from "@/components/Card";
import {
  BarChart3,
  Cpu,
  Layers,
  Scale,
  TrendingUp,
  UserCheck,
  ShieldCheck,
  Play,
  CheckCircle2,
  Loader2,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export default function ModelEvaluationPage() {
  const [activeDataset, setActiveDataset] = useState<"real" | "human" | "ats" | "synthetic">("real");
  const [metrics, setMetrics] = useState<ModelMetricsResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [humanVal, setHumanVal] = useState<HumanValidationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [runningBenchmark, setRunningBenchmark] = useState(false);
  const [benchmarkNotice, setBenchmarkNotice] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getModelMetrics("real"),
      api.getComparison(),
      api.getHumanValidation().catch(() => null),
    ])
      .then(([m, c, h]) => {
        setMetrics(m);
        setComparison(c);
        if (h) setHumanVal(h);
      })
      .catch((err) => console.error("Error loading metrics:", err))
      .finally(() => setLoading(false));
  }, []);

  const handleDatasetChange = async (type: "real" | "human" | "ats" | "synthetic") => {
    setActiveDataset(type);
    if (type === "human") {
      if (!humanVal) {
        setMetricsLoading(true);
        try {
          const data = await api.getHumanValidation();
          setHumanVal(data);
        } catch (err) {
          console.error("Error fetching human validation:", err);
        } finally {
          setMetricsLoading(false);
        }
      }
      return;
    }

    if (type === "ats") return;

    setMetricsLoading(true);
    try {
      const data = await api.getModelMetrics(type);
      setMetrics(data);
    } catch (err) {
      console.error("Error switching dataset:", err);
    } finally {
      setMetricsLoading(false);
    }
  };

  const handleRunLiveBenchmark = async () => {
    setRunningBenchmark(true);
    setBenchmarkNotice(null);
    try {
      const updated = await api.runModelBenchmark();
      setMetrics(updated);
      setBenchmarkNotice("Live benchmark pipeline executed successfully across 1,512 pairs with StratifiedGroupKFold.");
      setTimeout(() => setBenchmarkNotice(null), 6000);
    } catch (err: any) {
      setBenchmarkNotice(`Benchmark execution failed: ${err.message}`);
    } finally {
      setRunningBenchmark(false);
    }
  };

  if (loading) {
    return (
      <div className="py-20 text-center text-slate-500 text-xs">
        <div className="animate-spin w-8 h-8 border-2 border-slate-900 border-t-transparent rounded-full mx-auto mb-3" />
        Loading model benchmark metrics and validation datasets...
      </div>
    );
  }

  const selectedModel = metrics?.models.find((m) => m.name === metrics.selectedModel) || metrics?.models[0];
  const chartData = selectedModel
    ? Object.entries(selectedModel.featureImportance).map(([feat, val]) => ({
        feature: feat.replace(/_/g, " "),
        importance: val,
      }))
    : [];

  const devCalib = humanVal?.calibratedMetrics?.development_test_metrics;
  const humanCalib = humanVal?.calibratedMetrics?.human_validation_metrics;

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="pb-4 border-b border-slate-200">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center space-x-2">
              <BarChart3 className="w-6 h-6 text-slate-800" />
              <span>Model Benchmarks & Evaluation Diagnostics</span>
            </h1>
            <p className="text-xs text-slate-500 mt-1">
              Comparative empirical evaluation of candidate classification models (Logistic Regression, Random Forest, KNN) and calibration.
            </p>
          </div>

          <button
            onClick={handleRunLiveBenchmark}
            disabled={runningBenchmark}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-900 text-white hover:bg-slate-800 transition text-xs font-semibold shadow-xs disabled:opacity-50 self-start sm:self-auto"
          >
            {runningBenchmark ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Executing Pipeline...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Re-run Live Benchmark</span>
              </>
            )}
          </button>
        </div>

        {benchmarkNotice && (
          <div className="mt-3 p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>{benchmarkNotice}</span>
          </div>
        )}

        {/* Dataset Switcher Tabs */}
        <div className="flex flex-wrap items-center gap-2 mt-4 pt-3 border-t border-slate-100">
          <button
            onClick={() => handleDatasetChange("real")}
            disabled={metricsLoading}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition ${
              activeDataset === "real"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Production Benchmark (N=1,512 pairs, 126 resumes)</span>
          </button>

          <button
            onClick={() => handleDatasetChange("human")}
            disabled={metricsLoading}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition ${
              activeDataset === "human"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            <UserCheck className="w-3.5 h-3.5" />
            <span>Human Validation Benchmark (N=64 pairs, Single-Blind)</span>
          </button>

          <button
            onClick={() => handleDatasetChange("ats")}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition ${
              activeDataset === "ats"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5" />
            <span>Controlled ATS vs Semantic Comparison</span>
          </button>

          <button
            onClick={() => handleDatasetChange("synthetic")}
            disabled={metricsLoading}
            className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center space-x-2 transition ${
              activeDataset === "synthetic"
                ? "bg-slate-900 text-white shadow-xs"
                : "bg-slate-100 text-slate-700 hover:bg-slate-200"
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Synthetic Baseline (N=800 candidates)</span>
          </button>

          {metricsLoading && (
            <span className="text-xs text-slate-400 italic flex items-center space-x-1 ml-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-600 inline-block" />
              <span>Updating benchmark metrics...</span>
            </span>
          )}
        </div>
      </div>

      {/* 1. Production Canonical Benchmark View (Real & Synthetic) */}
      {(activeDataset === "real" || activeDataset === "synthetic") && (
        <div className="space-y-8">
          {/* Methodology Disclaimer Banner */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 text-xs flex items-start space-x-3">
            <Scale className="w-5 h-5 text-slate-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block">Methodological Note & Interpretation:</span>
              <p className="leading-relaxed text-slate-600">
                {metrics?.disclaimer ||
                  "Benchmark measures how well machine learning classifiers reproduce the defined qualification screening policy. StratifiedGroupKFold splitting by resume ID guarantees test resumes are never seen during training (zero data leakage)."}
              </p>
            </div>
          </div>

          {/* Dataset Specifications Card */}
          {metrics?.dataset && (
            <Card
              title="Validation Dataset Specifications"
              subtitle={metrics.dataset.source}
              icon={Layers}
            >
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                  <span className="text-slate-400 block text-[10px] uppercase font-sans">Total Records</span>
                  <span className="text-xl font-bold text-slate-900 mt-1 block">{metrics.dataset.records} pairs</span>
                </div>
                <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                  <span className="text-slate-400 block text-[10px] uppercase font-sans">Train / Test Partition</span>
                  <span className="text-xl font-bold text-slate-900 mt-1 block">
                    {metrics.dataset.trainRecords} / {metrics.dataset.testRecords} (80/20)
                  </span>
                </div>
                <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                  <span className="text-slate-400 block text-[10px] uppercase font-sans">Validation Scheme</span>
                  <span className="text-sm font-bold text-slate-900 mt-1 block truncate font-sans">
                    StratifiedGroupKFold
                  </span>
                </div>
                <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                  <span className="text-slate-400 block text-[10px] uppercase font-sans">Selected Architecture</span>
                  <span className="text-sm font-bold text-slate-900 mt-1 block truncate font-sans">
                    {metrics.selectedModel}
                  </span>
                </div>
              </div>
            </Card>
          )}

          {/* Model Comparison Table */}
          <Card
            title="Classification Model Performance Comparison (Test Set)"
            subtitle="Evaluated on Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Shortlist Recall, and 5-Fold Cross Validation"
            icon={Cpu}
          >
            <div className="overflow-x-auto -mx-5 -my-5">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-3 px-4 font-semibold">Model Architecture</th>
                    <th className="py-3 px-4 font-semibold text-center">Accuracy</th>
                    <th className="py-3 px-4 font-semibold text-center">Precision (Macro)</th>
                    <th className="py-3 px-4 font-semibold text-center">Recall (Macro)</th>
                    <th className="py-3 px-4 font-semibold text-center">Macro F1</th>
                    <th
                      className="py-3 px-4 font-semibold text-center text-slate-900"
                      title="Crucial in recruitment: detects suitable candidates without missing talent"
                    >
                      Shortlist Recall
                    </th>
                    <th className="py-3 px-4 font-semibold text-center">5-Fold CV F1 (Mean ± Std)</th>
                    <th className="py-3 px-4 font-semibold text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {metrics?.models.map((m) => {
                    const isSelected = m.name === metrics.selectedModel;
                    return (
                      <tr key={m.name} className={isSelected ? "bg-slate-50 font-semibold" : "hover:bg-slate-50/60"}>
                        <td className="py-3 px-4 font-sans font-semibold text-slate-900">
                          {m.name}
                        </td>
                        <td className="py-3 px-4 text-center">{(m.accuracy * 100).toFixed(1)}%</td>
                        <td className="py-3 px-4 text-center">{(m.precision * 100).toFixed(1)}%</td>
                        <td className="py-3 px-4 text-center">{(m.recall * 100).toFixed(1)}%</td>
                        <td className="py-3 px-4 text-center font-bold text-slate-900">
                          {(m.f1 * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-center font-bold text-slate-900">
                          {(m.shortlistRecall * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-center text-slate-600">
                          {(m.cvF1Mean * 100).toFixed(1)}% ± {(m.cvF1Std * 100).toFixed(1)}%
                        </td>
                        <td className="py-3 px-4 text-center font-sans">
                          {isSelected ? (
                            <span className="px-2.5 py-0.5 rounded bg-slate-900 text-white text-[10px] font-semibold">
                              SELECTED
                            </span>
                          ) : (
                            <span className="text-slate-400 text-[11px]">Evaluated</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Probability Calibration Comparison Table */}
          {metrics?.calibrationComparison && (
            <Card
              title="Probability Calibration Analysis on Held-Out Test Data"
              subtitle="Comparison of Uncalibrated Logistic Regression vs Platt Scaling vs Isotonic Calibration"
              icon={Scale}
            >
              <div className="overflow-x-auto -mx-5 -my-5">
                <table className="w-full text-xs text-left">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase tracking-wider text-[11px]">
                    <tr>
                      <th className="py-3 px-4 font-semibold">Calibration Method</th>
                      <th className="py-3 px-4 font-semibold text-center">Log Loss (Lower is Better)</th>
                      <th className="py-3 px-4 font-semibold text-center">Brier Score (Lower is Better)</th>
                      <th className="py-3 px-4 font-semibold text-center">Macro F1 Score</th>
                      <th className="py-3 px-4 font-semibold text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    {Object.entries(metrics.calibrationComparison).map(([method, data]: [string, any]) => {
                      const isSelected = method === "isotonic";
                      const label =
                        method === "uncalibrated"
                          ? "Uncalibrated (Raw Logistic)"
                          : method === "platt_sigmoid"
                          ? "Platt Scaling (Sigmoid)"
                          : "Isotonic Regression (Non-parametric)";
                      const logLoss = data.logLoss ?? data.log_loss ?? 0;
                      const brierScore = data.brierScore ?? data.brier_score ?? 0;
                      const macroF1 = data.macroF1 ?? data.macro_f1 ?? 0;
                      return (
                        <tr key={method} className={isSelected ? "bg-slate-50 font-semibold" : "hover:bg-slate-50/60"}>
                          <td className="py-3 px-4 font-sans font-semibold text-slate-900">{label}</td>
                          <td className="py-3 px-4 text-center font-bold text-slate-900">{Number(logLoss).toFixed(4)}</td>
                          <td className="py-3 px-4 text-center">{Number(brierScore).toFixed(4)}</td>
                          <td className="py-3 px-4 text-center">{(Number(macroF1) * 100).toFixed(2)}%</td>
                          <td className="py-3 px-4 text-center font-sans">
                            {isSelected ? (
                              <span className="px-2.5 py-0.5 rounded bg-emerald-600 text-white text-[10px] font-semibold">
                                SELECTED CALIBRATOR
                              </span>
                            ) : (
                              <span className="text-slate-400 text-[11px]">Evaluated</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Feature Importance & Confusion Matrix */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Feature Importance Chart */}
            <Card
              title={`Feature Contributions (${metrics?.selectedModel})`}
              subtitle={selectedModel?.importanceMethod || "Relative feature contribution to decision"}
            >
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 110, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                    <XAxis type="number" tick={{ fontSize: 10 }} />
                    <YAxis
                      dataKey="feature"
                      type="category"
                      tick={{ fontSize: 10, fill: "#475569" }}
                      width={110}
                    />
                    <Tooltip
                      formatter={(val: any) => [Number(val).toFixed(3), "Magnitude"]}
                      contentStyle={{ fontSize: 11, borderRadius: 8 }}
                    />
                    <Bar dataKey="importance" fill="#0f172a" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Confusion Matrix Heatmap */}
            <Card
              title={`Confusion Matrix (${metrics?.selectedModel})`}
              subtitle="Rows: Actual Label • Columns: Predicted Label"
            >
              {selectedModel && metrics?.dataset && (
                <div className="flex flex-col items-center justify-center py-4">
                  <div className="grid grid-cols-4 gap-1.5 text-center font-mono text-xs max-w-sm w-full">
                    <div className="p-2 font-bold text-slate-400 text-[10px]">Actual \ Pred</div>
                    {metrics.dataset.labels.map((l) => (
                      <div key={l} className="p-2 font-bold text-slate-600 uppercase text-[10px]">
                        {l.replace("_", " ")}
                      </div>
                    ))}

                    {selectedModel.confusionMatrix.map((row, rowIdx) => (
                      <React.Fragment key={rowIdx}>
                        <div className="p-2 font-bold text-slate-600 uppercase text-[10px] text-right self-center">
                          {metrics.dataset?.labels[rowIdx].replace("_", " ")}
                        </div>
                        {row.map((val, colIdx) => {
                          const isDiagonal = rowIdx === colIdx;
                          return (
                            <div
                              key={colIdx}
                              className={`p-3 rounded-lg border font-bold text-sm ${
                                isDiagonal
                                  ? "bg-emerald-50 text-emerald-900 border-emerald-200"
                                  : val > 0
                                  ? "bg-rose-50 text-rose-800 border-rose-200"
                                  : "bg-slate-50 text-slate-400 border-slate-200"
                              }`}
                            >
                              {val}
                            </div>
                          );
                        })}
                      </React.Fragment>
                    ))}
                  </div>
                  <p className="text-[11px] text-slate-400 mt-4 text-center">
                    Diagonal represents correct classifications; off-diagonal cells represent misclassifications.
                  </p>
                </div>
              )}
            </Card>
          </div>
        </div>
      )}

      {/* 2. Human Validation Tab Content */}
      {activeDataset === "human" && humanVal && (
        <div className="space-y-6">
          {/* Academic Notice Banner */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-slate-800 text-xs flex items-start space-x-3">
            <ShieldCheck className="w-5 h-5 text-slate-700 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-bold block">Human Validation Benchmark Protocol:</span>
              <p className="leading-relaxed text-slate-600">
                Single-blind evaluation over {humanVal.totalPairs} curated edge-case pairs evaluated independently by a human recruiter ({humanVal.reviewerId}) without visibility into model scores or rule weights.
              </p>
            </div>
          </div>

          {/* Metrics Counters */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Validation Pairs</span>
              <span className="text-2xl font-black font-mono text-slate-900 mt-1 block">{humanVal.totalPairs}</span>
              <span className="text-[11px] text-slate-500 mt-0.5 block">{humanVal.totalResumes} candidates × {humanVal.totalJds} roles</span>
            </div>

            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Shortlist Precision</span>
              <span className="text-2xl font-black font-mono text-emerald-600 mt-1 block">{(humanVal.mlVsHuman.shortlistPrecision * 100).toFixed(1)}%</span>
              <span className="text-[11px] text-slate-500 mt-0.5 block">High purity (low false shortlists)</span>
            </div>

            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Shortlist Recall</span>
              <span className="text-2xl font-black font-mono text-amber-600 mt-1 block">{(humanVal.mlVsHuman.shortlistRecall * 100).toFixed(1)}%</span>
              <span className="text-[11px] text-slate-500 mt-0.5 block">Conservative screening threshold</span>
            </div>

            <div className="p-4 bg-white rounded-xl border border-slate-200 shadow-xs">
              <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Reviewer Protocol</span>
              <span className="text-2xl font-black font-mono text-slate-900 mt-1 block">Single-Blind</span>
              <span className="text-[11px] text-slate-500 mt-0.5 block">Reviewer ID: {humanVal.reviewerId}</span>
            </div>
          </div>

          {/* Confusion Matrix & Calibration Impact */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Human Confusion Matrix (3x3)" subtitle="Human Ground Truth (Rows) vs Model Prediction (Columns)">
              <div className="overflow-x-auto text-xs">
                <table className="w-full text-center border-collapse">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-500 text-[10px] uppercase font-bold">
                      <th className="py-2 text-left">Human \ Model</th>
                      <th className="py-2 px-3 bg-slate-50 text-slate-700">Pred: Weak</th>
                      <th className="py-2 px-3 bg-amber-50 text-amber-700">Pred: Review</th>
                      <th className="py-2 px-3 bg-emerald-50 text-emerald-700">Pred: Shortlist</th>
                      <th className="py-2 px-3">Recall</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 font-mono">
                    <tr>
                      <td className="py-2 text-left font-sans font-semibold">True: Weak Match</td>
                      <td className="py-2 px-3 font-bold text-emerald-600 bg-emerald-50/20">{humanVal.mlVsHuman.confusionMatrix[0]?.[0]}</td>
                      <td className="py-2 px-3 text-slate-600">{humanVal.mlVsHuman.confusionMatrix[0]?.[1]}</td>
                      <td className="py-2 px-3 text-rose-600 font-semibold">{humanVal.mlVsHuman.confusionMatrix[0]?.[2]}</td>
                      <td className="py-2 px-3 text-slate-500">{((humanVal.mlVsHuman.perClass?.weak_match?.recall || 0) * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                      <td className="py-2 text-left font-sans font-semibold">True: Review</td>
                      <td className="py-2 px-3 text-slate-600">{humanVal.mlVsHuman.confusionMatrix[1]?.[0]}</td>
                      <td className="py-2 px-3 font-bold text-emerald-600 bg-emerald-50/20">{humanVal.mlVsHuman.confusionMatrix[1]?.[1]}</td>
                      <td className="py-2 px-3 text-slate-600">{humanVal.mlVsHuman.confusionMatrix[1]?.[2]}</td>
                      <td className="py-2 px-3 text-slate-500">{((humanVal.mlVsHuman.perClass?.review?.recall || 0) * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                      <td className="py-2 text-left font-sans font-semibold">True: Shortlist</td>
                      <td className="py-2 px-3 text-rose-600 font-semibold">{humanVal.mlVsHuman.confusionMatrix[2]?.[0]}</td>
                      <td className="py-2 px-3 text-slate-600">{humanVal.mlVsHuman.confusionMatrix[2]?.[1]}</td>
                      <td className="py-2 px-3 font-bold text-emerald-600 bg-emerald-50/20">{humanVal.mlVsHuman.confusionMatrix[2]?.[2]}</td>
                      <td className="py-2 px-3 text-slate-500">{((humanVal.mlVsHuman.perClass?.shortlist?.recall || 0) * 100).toFixed(1)}%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </Card>

            {/* Dynamic Calibration Impact Table */}
            <Card title="Calibration Impact on Human Validation" subtitle="Evaluated across uncalibrated, class-weighted, and calibrated models">
              {devCalib && humanCalib ? (
                <div className="overflow-x-auto text-xs font-mono">
                  <table className="w-full text-left">
                    <thead className="bg-slate-50 text-slate-500 text-[10px] uppercase font-bold border-b border-slate-200">
                      <tr>
                        <th className="py-2 px-3">Configuration</th>
                        <th className="py-2 px-3 text-center">Dev Acc</th>
                        <th className="py-2 px-3 text-center">Human Acc</th>
                        <th className="py-2 px-3 text-center">Shortlist Recall</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      <tr>
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">A. Unweighted Baseline</td>
                        <td className="py-2 px-3 text-center">{(devCalib.baseline_unweighted.accuracy * 100).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-center">{(humanCalib.baseline_unweighted_on_human.accuracy * 100).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-center font-bold text-slate-700">{(humanCalib.baseline_unweighted_on_human.shortlist_recall * 100).toFixed(2)}%</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">B. Class-Weighted</td>
                        <td className="py-2 px-3 text-center">{(devCalib.class_weighted.accuracy * 100).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-center">{(humanCalib.class_weighted_on_human.accuracy * 100).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-center font-bold text-slate-700">{(humanCalib.class_weighted_on_human.shortlist_recall * 100).toFixed(2)}%</td>
                      </tr>
                      <tr className="bg-slate-50">
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">C. Calibrated Model (Isotonic)</td>
                        <td className="py-2 px-3 text-center">{(devCalib.calibrated_model.accuracy * 100).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-center font-bold text-slate-900">{(humanCalib.calibrated_model_on_human.accuracy * 100).toFixed(2)}%</td>
                        <td className="py-2 px-3 text-center font-bold text-emerald-600">{(humanCalib.calibrated_model_on_human.shortlist_recall * 100).toFixed(2)}%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-slate-400">Calibration metrics not available.</div>
              )}
            </Card>
          </div>

          {/* Human Labeled Pairs Explorer */}
          <Card title="Human Validation Pairs Explorer (N=64)" subtitle="Single-blind recruiter labels, evidence, and notes across all 12 job descriptions">
            <div className="overflow-x-auto -mx-5 -my-5">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase tracking-wider text-[11px]">
                  <tr>
                    <th className="py-3 px-4 font-semibold">Candidate</th>
                    <th className="py-3 px-4 font-semibold">Job Title</th>
                    <th className="py-3 px-4 font-semibold text-center">Human Label</th>
                    <th className="py-3 px-4 font-semibold text-center">Model Prediction</th>
                    <th className="py-3 px-4 font-semibold text-center">Agreement</th>
                    <th className="py-3 px-4 font-semibold">Recruiter Notes & Evidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {humanVal.pairs.map((p, idx) => {
                    const agreed =
                      (p.humanLabel === "STRONG_MATCH" || p.humanLabel === "MATCH")
                        ? p.mlPrediction === "shortlist"
                        : p.humanLabel === "REVIEW"
                        ? p.mlPrediction === "review"
                        : p.mlPrediction === "weak_match";
                    return (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="py-2.5 px-4 font-semibold text-slate-900">{p.candidateName}</td>
                        <td className="py-2.5 px-4 text-slate-600">{p.jobTitle}</td>
                        <td className="py-2.5 px-4 text-center font-mono font-bold">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] ${
                              p.humanLabel.includes("MATCH")
                                ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                                : p.humanLabel === "REVIEW"
                                ? "bg-amber-50 text-amber-800 border border-amber-200"
                                : "bg-rose-50 text-rose-800 border border-rose-200"
                            }`}
                          >
                            {p.humanLabel}
                          </span>
                        </td>
                        <td className="py-2.5 px-4 text-center font-mono">
                          <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-700 text-[10px]">
                            {p.mlPrediction}
                          </span>
                        </td>
                        <td className="py-2.5 px-4 text-center">
                          {agreed ? (
                            <span className="text-emerald-700 font-semibold text-[11px]">✓ Agreed</span>
                          ) : (
                            <span className="text-amber-700 font-semibold text-[11px]">⚠ Disagreed</span>
                          )}
                        </td>
                        <td className="py-2.5 px-4 text-slate-600 text-[11px] max-w-xs">
                          <p className="line-clamp-2">{p.notes}</p>
                          <span className="text-[10px] text-slate-400 italic block mt-0.5 truncate">
                            Evidence: {p.evidence}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}

      {/* 3. Controlled ATS vs Semantic Comparison View */}
      {activeDataset === "ats" && (
        <Card
          title="Controlled Empirical Comparison: Keyword ATS vs Semantic Embeddings"
          subtitle="Empirical proof demonstrating why semantic transformer embeddings outperform keyword scanners across edge cases"
          icon={TrendingUp}
        >
          <div className="space-y-4">
            {comparison?.cases.map((c) => (
              <div
                key={c.caseId}
                className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 space-y-3 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-900 text-sm">{c.title}</span>
                  <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-800 font-mono text-[10px]">
                    Requirement: {c.requirement}
                  </span>
                </div>

                <p className="text-slate-600">{c.description}</p>

                <div className="p-2.5 rounded-lg bg-white border border-slate-200 font-mono text-[11px] text-slate-700">
                  <span className="text-slate-400 block text-[9px] uppercase font-sans">Resume Excerpt:</span>
                  &ldquo;{c.resumeText}&rdquo;
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
                  <div className="p-3 rounded-lg bg-rose-50/60 border border-rose-200 space-y-1">
                    <span className="font-bold text-rose-900 text-[11px] uppercase block">
                      Traditional Keyword ATS:
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-rose-800">
                        {c.keywordAtsFound ? "✓ Term Found" : "✗ Term Missing"}
                      </span>
                      <span className="text-slate-500 font-mono">({c.keywordMentions} mentions)</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-emerald-50/60 border border-emerald-200 space-y-1">
                    <span className="font-bold text-emerald-900 text-[11px] uppercase block">
                      Semantic AI System:
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold text-emerald-800">
                        Status: {c.aiStatus}
                      </span>
                      <span className="text-slate-500 font-mono">
                        (Sim: {c.aiSimilarity.toFixed(2)})
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-2.5 rounded-lg bg-white border border-slate-200 text-slate-800 font-medium">
                  {c.verdict}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
