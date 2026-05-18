import { defineStore } from 'pinia';
import { api } from '@/api/client';
import { NUM_DEFECT_TYPES } from '@/api/types';
import type { CurvePoint, ThresholdResponse } from '@/api/types';

type Approach = 'attention' | 'seq2seq';

export const useThresholdStore = defineStore('threshold', {
  state: () => ({
    approach: 'attention' as Approach,
    thresholds: Array.from({ length: NUM_DEFECT_TYPES }, () => 0.5),
    metrics: null as ThresholdResponse | null,
    curves: null as CurvePoint[][] | null,
    loading: false,
  }),
  actions: {
    async refreshMetrics() {
      this.loading = true;
      try {
        this.metrics = await api.evaluateThresholds(this.thresholds, this.approach);
      } finally {
        this.loading = false;
      }
    },
    async refreshCurves() {
      const response = await api.thresholdCurves(this.approach);
      this.curves = response.per_class_curves;
    },
    async setThreshold(index: number, value: number) {
      this.thresholds = this.thresholds.map((current, i) => (i === index ? value : current));
      await this.refreshMetrics();
    },
    async setApproach(approach: Approach) {
      this.approach = approach;
      await Promise.all([this.refreshMetrics(), this.refreshCurves()]);
    },
  },
});
