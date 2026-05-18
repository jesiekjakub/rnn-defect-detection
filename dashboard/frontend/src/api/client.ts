import type {
  ArchitectureResponse,
  AttentionResponse,
  BatchResponse,
  CompareResponse,
  HealthResponse,
  LatentPoint,
  MetricsResponse,
  SampleSpec,
  Seq2SeqResponse,
  Sequence,
  StreamSnapshot,
  ThresholdCurves,
  ThresholdResponse,
} from './types';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '';

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

export const api = {
  health(): Promise<HealthResponse> {
    return fetchJson<HealthResponse>('/health');
  },
  generateSample(spec: SampleSpec): Promise<Sequence> {
    return fetchJson<Sequence>('/api/sample', {
      method: 'POST',
      body: JSON.stringify(spec),
    });
  },
  predictAttention(sequence: Sequence): Promise<AttentionResponse> {
    return fetchJson<AttentionResponse>('/api/predict/attention', {
      method: 'POST',
      body: JSON.stringify(sequence),
    });
  },
  predictSeq2Seq(sequence: Sequence): Promise<Seq2SeqResponse> {
    return fetchJson<Seq2SeqResponse>('/api/predict/seq2seq', {
      method: 'POST',
      body: JSON.stringify(sequence),
    });
  },
  predictCompare(sequence: Sequence): Promise<CompareResponse> {
    return fetchJson<CompareResponse>('/api/predict/compare', {
      method: 'POST',
      body: JSON.stringify(sequence),
    });
  },
  predictStream(sequence: Sequence, windowSize: number, stride: number): Promise<{ snapshots: StreamSnapshot[] }> {
    return fetchJson<{ snapshots: StreamSnapshot[] }>('/api/predict/stream', {
      method: 'POST',
      body: JSON.stringify({ sequence, window_size: windowSize, stride }),
    });
  },
  evaluateThresholds(thresholds: number[], approach: 'attention' | 'seq2seq'): Promise<ThresholdResponse> {
    return fetchJson<ThresholdResponse>('/api/threshold/evaluate', {
      method: 'POST',
      body: JSON.stringify({ thresholds, approach }),
    });
  },
  thresholdCurves(approach: 'attention' | 'seq2seq'): Promise<ThresholdCurves> {
    return fetchJson<ThresholdCurves>(`/api/threshold/curves?approach=${approach}`);
  },
  latent(): Promise<{ points: LatentPoint[] }> {
    return fetchJson<{ points: LatentPoint[] }>('/api/latent');
  },
  latentSample(sampleId: number): Promise<CompareResponse> {
    return fetchJson<CompareResponse>(`/api/latent/sample/${sampleId}`);
  },
  batch(params: {
    offset: number;
    limit: number;
    predDefect?: number;
    confidenceMin?: number;
    confidenceMax?: number;
    agreementOnly?: boolean;
    approach?: 'attention' | 'seq2seq';
  }): Promise<BatchResponse> {
    const query = new URLSearchParams();
    query.set('offset', String(params.offset));
    query.set('limit', String(params.limit));
    if (params.predDefect !== undefined) query.set('pred_defect', String(params.predDefect));
    if (params.confidenceMin !== undefined) query.set('confidence_min', String(params.confidenceMin));
    if (params.confidenceMax !== undefined) query.set('confidence_max', String(params.confidenceMax));
    if (params.agreementOnly) query.set('agreement_only', 'true');
    if (params.approach) query.set('approach', params.approach);
    return fetchJson<BatchResponse>(`/api/batch?${query.toString()}`);
  },
  batchSample(sampleId: number): Promise<CompareResponse> {
    return fetchJson<CompareResponse>(`/api/batch/${sampleId}`);
  },
  metrics(): Promise<MetricsResponse> {
    return fetchJson<MetricsResponse>('/api/metrics');
  },
  architecture(): Promise<ArchitectureResponse> {
    return fetchJson<ArchitectureResponse>('/api/architecture');
  },
  uploadFile(file: File): Promise<{ sequences: Sequence[]; warnings: string[] }> {
    const form = new FormData();
    form.append('file', file);
    return fetch(`${BASE}/api/upload`, { method: 'POST', body: form }).then(async (response) => {
      if (!response.ok) throw new ApiError(response.status, await response.text());
      return response.json();
    });
  },
};
