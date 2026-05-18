import { defineStore } from 'pinia';
import { api } from '@/api/client';
import type { BatchResponse } from '@/api/types';

export const useBatchStore = defineStore('batch', {
  state: () => ({
    rows: null as BatchResponse | null,
    offset: 0,
    limit: 50,
    predDefect: undefined as number | undefined,
    confidenceMin: 0,
    confidenceMax: 1,
    agreementOnly: false,
    loading: false,
  }),
  actions: {
    async refresh() {
      this.loading = true;
      try {
        this.rows = await api.batch({
          offset: this.offset,
          limit: this.limit,
          predDefect: this.predDefect,
          confidenceMin: this.confidenceMin,
          confidenceMax: this.confidenceMax,
          agreementOnly: this.agreementOnly,
        });
      } finally {
        this.loading = false;
      }
    },
    setFilters(partial: Partial<{
      predDefect: number | undefined;
      confidenceMin: number;
      confidenceMax: number;
      agreementOnly: boolean;
    }>) {
      Object.assign(this, partial);
      this.offset = 0;
    },
  },
});
