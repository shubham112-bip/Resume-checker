/**
 * TypeScript contracts matching the FastAPI backend camelCase responses.
 */

export type Recommendation = "STRONG_SHORTLIST" | "SHORTLIST" | "REVIEW" | "WEAK_MATCH";

export type EvidenceStatus =
  | "MATCH"
  | "SEMANTIC_MATCH"
  | "CLAIMED_ONLY"
  | "INFERRED"
  | "UNCERTAIN"
  | "NOT_FOUND";

export type SkillTier = "REQUIRED" | "PREFERRED" | "OPTIONAL";

export interface ContactInfo {
  email: string | null;
  phone: string | null;
  linkedin: string | null;
  github: string | null;
  portfolio: string | null;
}

export interface EducationEntry {
  degree: string;
  degreeLevel: number;
  field: string | null;
  isRelevantField: boolean;
  institution: string | null;
  year: string | null;
  raw: string;
}

export interface ExperienceEntry {
  title: string;
  organization: string;
  start: string | null;
  end: string | null;
  durationMonths: number;
  isCurrent: boolean;
  isInternship: boolean;
  bullets: string[];
}

export interface ProjectEntry {
  name: string;
  description: string;
  technologies: string[];
}

export interface DocumentDiagnostics {
  fileType: string;
  parserUsed: string;
  pageCount: number;
  charCount: number;
  wordCount: number;
  imageCount: number;
  tableCount: number;
  isScanned: boolean;
  warnings: string[];
}

export interface SensitiveAttributeFinding {
  attribute: string;
  action: string;
}

export interface ResumeSection {
  name: string;
  heading: string;
  text: string;
  startLine: number;
}

export interface CandidateProfile {
  candidateId: string;
  name: string;
  contact: ContactInfo;
  summary: string;
  education: EducationEntry[];
  skills: string[];
  experience: ExperienceEntry[];
  projects: ProjectEntry[];
  certifications: string[];
  achievements: string[];
  sections: ResumeSection[];
  unrecognizedHeadings: string[];
  totalExperienceMonths: number;
  diagnostics: DocumentDiagnostics | null;
  sensitiveAttributesRemoved: SensitiveAttributeFinding[];
}

export interface JobSkill {
  name: string;
  tier: SkillTier;
  category: string;
  context: string;
}

export interface JobProfile {
  jobId: string;
  title: string;
  skills: JobSkill[];
  minExperienceYears: number;
  maxExperienceYears: number | null;
  educationRequirements: string[];
  requiredDegreeLevel: number;
  requiresRelevantField: boolean;
  responsibilities: string[];
  tools: string[];
  domainRequirements: string[];
  softSkills: string[];
  extractionMethod: string;
  rawText: string;
}

export interface EvidencePassage {
  text: string;
  section: string;
  similarity: number;
  matchType: string;
}

export interface RequirementMatch {
  requirement: string;
  tier: SkillTier;
  status: EvidenceStatus;
  confidence: number;
  credit: number;
  keywordFound: boolean;
  reflected: boolean;
  evidence: EvidencePassage[];
  note: string;
}

export interface ATSCheck {
  name: string;
  category: string;
  passed: boolean;
  detail: string;
}

export interface ATSResult {
  score: number;
  contactScore: number;
  sectionScore: number;
  keywordCoverage: number;
  formattingScore: number;
  checks: ATSCheck[];
  issues: string[];
  matchedKeywords: string[];
  missingKeywords: string[];
}

export interface StuffedTerm {
  term: string;
  mentions: number;
  evidenceMentions: number;
}

export interface StuffingResult {
  suspected: boolean;
  keywordDensity: number;
  flaggedTerms: StuffedTerm[];
  detail: string;
}

export interface FeatureVector {
  requiredSkillMatch: number;
  preferredSkillMatch: number;
  experienceMatch: number;
  educationMatch: number;
  projectMatch: number;
  semanticSimilarity: number;
  atsScore: number;
  resumeQuality: number;
}

export interface ScoreComponent {
  key: string;
  label: string;
  weight: number;
  value: number;
  contribution: number;
}

export interface ScoreBreakdown {
  components: ScoreComponent[];
  overallScore: number;
  formula: string;
}

export interface MLPrediction {
  available: boolean;
  modelName: string;
  predictedLabel: string;
  probabilities: Record<string, number>;
  agreesWithRules: boolean;
  detail: string;
  modelVersion?: string;
  featureVersion?: string;
  featureContributions?: Record<string, number>;
}

export interface HybridDecision {
  ruleScore: number;
  ruleDecision: string;
  mlDecision: string;
  mlProbability: number;
  finalDecision: string;
  agreement: boolean;
  isBorderline: boolean;
  policyReason: string;
  reviewRecommendationFlag?: string | null;
  featureContributions?: Record<string, number>;
}

export interface ReviewFlag {
  code: string;
  message: string;
}

export interface AgentTraceEvent {
  step: number;
  agent: string;
  task: string;
  tool: string | null;
  toolInput: string | null;
  observation: string;
  status: "success" | "warning" | "error" | "skipped";
  durationMs: number;
  details: Record<string, unknown>;
  timestamp: string;
}

export interface Explanation {
  summary: string;
  strengths: string[];
  weaknesses: string[];
  rationale: string;
  generatedBy: string;
  model: string | null;
  fallbackReason: string | null;
}

export interface MemoryEpisode {
  candidateName: string;
  jobTitle: string;
  overallScore: number;
  recommendation: string;
  createdAt: string;
}

export interface CandidateEvaluation {
  evaluationId: string;
  runId: string;
  candidateId: string;
  jobId: string;
  jobTitle: string;
  candidateName: string;
  fileName: string;
  status: string;
  error: string | null;
  overallScore: number;
  atsScore: number;
  requiredSkillMatch: number;
  preferredSkillMatch: number;
  experienceMatch: number;
  semanticSimilarity: number;
  projectMatch: number;
  educationMatch: number;
  resumeQuality: number;
  qualityEvaluation?: ResumeQualityEvaluation | null;
  keywordAtsScore: number;
  scoreBreakdown: ScoreBreakdown | null;
  features: FeatureVector | null;
  matchedSkills: string[];
  missingSkills: string[];
  uncertainSkills: string[];
  requirementMatches: RequirementMatch[];
  ats: ATSResult | null;
  stuffing: StuffingResult | null;
  mlPrediction: MLPrediction | null;
  hybridDecision?: HybridDecision | null;
  recommendation: Recommendation;
  recommendationLabel: string;
  needsHumanReview: boolean;
  reviewFlags: ReviewFlag[];
  strengths: string[];
  weaknesses: string[];
  explanation: Explanation | null;
  experienceYears: number;
  profile: CandidateProfile | null;
  recruiterDecision?: "SHORTLIST" | "REVIEW" | "REJECT" | null;
  recruiterNotes?: string;
  recruiterDecidedAt?: string;
  memoryRecall: MemoryEpisode[];
  trace: AgentTraceEvent[];
  createdAt: string;
}

export interface RankingEntry {
  rank: number;
  evaluationId: string;
  candidateId: string;
  candidateName: string;
  fileName: string;
  status: string;
  overallScore: number;
  recommendation: Recommendation;
  recommendationLabel: string;
  needsHumanReview: boolean;
  reviewFlags: string[];
  keywordAtsScore: number;
  keywordRank: number;
  mlLabel: string | null;
  matchedSkills: string[];
  missingSkills: string[];
}

export interface EvaluationRunResponse {
  runId: string;
  jobId: string;
  jobTitle: string;
  createdAt: string;
  durationMs: number;
  ranking: RankingEntry[];
  evaluations: CandidateEvaluation[];
}

export interface RunSummary {
  runId: string;
  jobId: string;
  jobTitle: string;
  candidateCount: number;
  createdAt: string;
}

export interface UploadedResume {
  candidateId: string;
  fileName: string;
  candidateName: string;
  status: string;
  warnings: string[];
}

export interface UploadError {
  fileName: string;
  error: string;
}

export interface UploadResponse {
  candidates: UploadedResume[];
  errors: UploadError[];
}

export interface ModelMetrics {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  shortlistRecall: number;
  cvF1Mean: number;
  cvF1Std: number;
  confusionMatrix: number[][];
  featureImportance: Record<string, number>;
  importanceMethod: string;
}

export interface DatasetInfo {
  source: string;
  records: number;
  trainRecords: number;
  testRecords: number;
  features: string[];
  target: string;
  labels: string[];
  classDistribution: Record<string, number>;
  preprocessing: string;
  split: string;
  randomSeed: number;
}

export interface CalibrationMethodMetrics {
  logLoss: number;
  brierScore: number;
  macroF1: number;
}

export interface ModelMetricsResponse {
  available: boolean;
  selectedModel: string | null;
  selectionCriterion: string;
  dataset: DatasetInfo | null;
  models: ModelMetrics[];
  disclaimer: string;
  trainedAt: string | null;
  calibrationComparison?: Record<string, CalibrationMethodMetrics> | null;
}

export interface ComparisonCase {
  caseId: string;
  title: string;
  description: string;
  requirement: string;
  resumeText: string;
  keywordAtsFound: boolean;
  keywordMentions: number;
  aiStatus: string;
  aiSimilarity: number;
  aiEvidence: string | null;
  stuffingSuspected: boolean;
  verdict: string;
}

export interface ComparisonResponse {
  cases: ComparisonCase[];
  embeddingBackend: string;
}

export interface HealthResponse {
  status: string;
  engine: string;
  embeddingBackend: string;
  embeddingModel: string;
  mlModelAvailable: boolean;
  selectedModel: string | null;
}

export interface DemoJob {
  demoId: string;
  title: string;
  text: string;
}

export interface DemoLoadResponse {
  job: JobProfile;
  upload: UploadResponse;
}

export interface CandidateSummary {
  candidateId: string;
  candidateName: string;
  fileName: string;
  createdAt: string;
  latestEvaluationId?: string | null;
  latestScore?: number | null;
  latestRecommendation?: string | null;
}

export interface EvaluationBrief {
  evaluationId: string;
  runId: string;
  jobTitle: string;
  overallScore: number;
  recommendation: Recommendation;
  createdAt: string;
}

export interface CandidateDetail {
  candidateId: string;
  candidateName: string;
  fileName: string;
  createdAt: string;
  profile: CandidateProfile | null;
  evaluations: EvaluationBrief[];
}

export interface EpisodeResponse {
  episodeId: number;
  candidateId: string;
  candidateName: string;
  jobId: string;
  jobTitle: string;
  eventType: string;
  summary: string;
  overallScore: number | null;
  createdAt: string;
}

export interface GraphResponse {
  mermaid: string;
  nodes: string[];
  conditionalEdges: string[];
}

export interface QualityDimension {
  name: string;
  weight: number;
  score: number;
  grade: string;
  feedback: string[];
}

export interface ResumeQualityEvaluation {
  overallQualityScore: number;
  qualityGrade: string;
  completenessScore: number;
  atsReadabilityScore: number;
  structureScore: number;
  actionVerbScore: number;
  quantificationScore: number;
  clarityScore: number;
  formattingConsistencyScore: number;
  dimensions: QualityDimension[];
  actionVerbsFound: string[];
  quantifiedBulletsCount: number;
  totalBulletsCount: number;
  strengths: string[];
  improvementAreas: string[];
  verdict: string;
}

export interface DatasetResumeBrief {
  resumeId: string;
  candidateName: string;
  fileType: string;
  wordCount: number;
  qualityScore: number;
  qualityGrade: string;
  actionVerbsCount: number;
  quantifiedBulletsCount: number;
}

export interface DatasetJdBrief {
  jdId: string;
  title: string;
  experienceLevel: string;
  requiredSkillsCount: number;
  preferredSkillsCount: number;
}

export interface DatasetSummaryResponse {
  totalResumes: number;
  totalJobDescriptions: number;
  totalPairs: number;
  classDistribution: Record<string, number>;
  labelsDistribution: Record<string, number>;
  labelSources: Record<string, number>;
  resumes: DatasetResumeBrief[];
  jobDescriptions: DatasetJdBrief[];
  notice: string;
}

export interface DatasetPairSummary {
  resumeId: string;
  jdId: string;
  candidateName: string;
  jobTitle: string;
  overallCandidateFitScore: number;
  resumeQualityScore: number;
  label: string;
  classificationLabel: string;
  labelSource: string;
  requiredSkillMatch: number;
  experienceMatch: number;
  semanticSimilarity: number;
  educationMatch: number;
  projectMatch: number;
  atsScore: number;
  resumeQuality: number;
}

export interface ResumeQualityUploadResponse {
  candidateName: string;
  fileName: string;
  evaluation: ResumeQualityEvaluation;
}

export interface HumanValidationPair {
  resumeId: string;
  jdId: string;
  candidateName: string;
  jobTitle: string;
  humanLabel: string;
  ruleLabel: string;
  mlPrediction: string;
  overallFitScore: number;
  resumeQualityScore: number;
  notes: string;
  evidence: string;
}

export interface HumanValidationResponse {
  available: boolean;
  totalPairs: number;
  totalResumes: number;
  totalJds: number;
  reviewerCount: number;
  reviewerId: string;
  classDistribution4class: Record<string, number>;
  classDistribution3class: Record<string, number>;
  mlVsHuman: {
    accuracy: number;
    macroPrecision: number;
    macroRecall: number;
    macroF1: number;
    weightedF1: number;
    shortlistPrecision: number;
    shortlistRecall: number;
    falseShortlistRate: number;
    confusionMatrix: number[][];
    labels: string[];
    perClass: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  };
  ruleVsHuman: {
    accuracy: number;
    macroPrecision: number;
    macroRecall: number;
    macroF1: number;
    weightedF1: number;
    shortlistPrecision: number;
    shortlistRecall: number;
    falseShortlistRate: number;
    confusionMatrix: number[][];
    labels: string[];
    perClass: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  };
  calibratedMetrics?: any;
  pairs: HumanValidationPair[];
  notice: string;
}

