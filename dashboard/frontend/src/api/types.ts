export type DefectIndex = 0 | 1 | 2 | 3 | 4;

export const DEFECT_NAMES = ['Spike S0', 'Dip S1', 'Zero S2', 'Offset S1', 'Pattern S0+S2'] as const;
export const DEFECT_COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f97316', '#a855f7'] as const;
export const SENSOR_COLORS = ['#60a5fa', '#34d399', '#fbbf24'] as const;
export const NUM_DEFECT_TYPES = 5;
export const NUM_SENSORS = 3;

export interface SampleSpec {
  defects: boolean[];
  seq_len: number;
  seed: number | null;
  noise_scale: number;
}

export interface Sequence {
  x: number[][];
  y_true: number[] | null;
  sequence_id?: string | null;
  origin: 'synthetic' | 'upload' | 'batch' | 'latent';
}

export interface AttentionExplanation {
  defect_index: number;
  defect_name: string;
  confidence: number;
  important_timesteps: number[];
  ranges: [number, number][];
  sensor_importance: number[];
}

export interface AttentionResponse {
  probs: number[];
  attention: number[][];
  explanations: AttentionExplanation[];
}

export interface CandidateRegion {
  start: number;
  end: number;
  source: 'residual' | 'velocity';
}

export interface VerifiedRegion {
  start: number;
  end: number;
  defect_index: number;
  defect_name: string;
  local_probability: number;
  consensus_pass: boolean;
}

export interface Seq2SeqResponse {
  probs: number[];
  reconstructed: number[][];
  residual: number[][];
  velocity: number[][];
  candidates: CandidateRegion[];
  verified: VerifiedRegion[];
  accepted_regions: VerifiedRegion[];
}

export interface CompareResponse {
  sequence: Sequence;
  attention: AttentionResponse;
  seq2seq: Seq2SeqResponse;
  agreement: boolean[];
}

export interface StreamSnapshot {
  t: number;
  probs_a1: number[];
  probs_a2: number[];
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  true_positive: number;
  false_positive: number;
  false_negative: number;
  true_negative: number;
}

export interface ThresholdResponse {
  per_class: PerClassMetric[];
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
}

export interface CurvePoint {
  threshold: number;
  precision: number;
  recall: number;
  fpr: number;
  tpr: number;
}

export interface ThresholdCurves {
  per_class_curves: CurvePoint[][];
}

export interface LatentPoint {
  sample_id: number;
  x: number;
  y: number;
  y_true: number[];
  y_pred_a1: number[];
  y_pred_a2: number[];
  agreement: boolean;
}

export interface BatchRow {
  sample_id: number;
  y_true: number[];
  probs_a1: number[];
  probs_a2: number[];
  pred_a1: number[];
  pred_a2: number[];
  agreement: boolean;
}

export interface BatchResponse {
  total: number;
  offset: number;
  limit: number;
  rows: BatchRow[];
}

export interface HealthResponse {
  status: 'ok' | 'warming';
  models_loaded: boolean;
  cache_ready: boolean;
  device: string;
}

export interface MetricsResponse {
  attention: Record<string, unknown> | null;
  seq2seq: Record<string, unknown> | null;
}

export interface ArchitectureLayer {
  name: string;
  kind: string;
  params: Record<string, unknown>;
}

export interface ArchitectureNode {
  id: string;
  label: string;
  layers: ArchitectureLayer[];
  notes: string | null;
}

export interface ArchitectureEdge {
  src: string;
  dst: string;
  label?: string | null;
}

export interface ArchitectureGraph {
  name: string;
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
}

export interface ArchitectureResponse {
  approaches: ArchitectureGraph[];
}
