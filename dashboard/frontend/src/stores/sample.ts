import { defineStore } from 'pinia';
import { api, ApiError } from '@/api/client';
import type { CompareResponse, SampleSpec, Sequence } from '@/api/types';
import { NUM_DEFECT_TYPES } from '@/api/types';

export const useSampleStore = defineStore('sample', {
  state: () => ({
    spec: {
      defects: [true, false, false, false, false],
      seq_len: 50,
      seed: 42,
      noise_scale: 0.2,
    } as SampleSpec,
    sequence: null as Sequence | null,
    compare: null as CompareResponse | null,
    loading: false,
    error: null as string | null,
  }),

  actions: {
    async generateAndPredict() {
      this.loading = true;
      this.error = null;
      try {
        const sequence = await api.generateSample(this.spec);
        const compare = await api.predictCompare(sequence);
        this.sequence = compare.sequence;
        this.compare = compare;
      } catch (err) {
        this.error = err instanceof ApiError ? err.message : (err as Error).message;
      } finally {
        this.loading = false;
      }
    },

    async loadSequence(sequence: Sequence) {
      this.loading = true;
      this.error = null;
      try {
        const compare = await api.predictCompare(sequence);
        this.sequence = compare.sequence;
        this.compare = compare;
      } catch (err) {
        this.error = err instanceof ApiError ? err.message : (err as Error).message;
      } finally {
        this.loading = false;
      }
    },

    async loadPrecomputed(compare: CompareResponse) {
      this.sequence = compare.sequence;
      this.compare = compare;
    },

    toggleDefect(index: number) {
      if (index < 0 || index >= NUM_DEFECT_TYPES) return;
      const next = [...this.spec.defects];
      next[index] = !next[index];
      this.spec.defects = next;
    },

    setSpec(partial: Partial<SampleSpec>) {
      this.spec = { ...this.spec, ...partial };
    },

    surpriseMe() {
      this.spec = {
        defects: Array.from({ length: NUM_DEFECT_TYPES }, () => Math.random() < 0.4),
        seq_len: 40 + Math.floor(Math.random() * 20),
        seed: Math.floor(Math.random() * 10_000),
        noise_scale: 0.1 + Math.random() * 0.3,
      };
    },
  },
});
