import { defineStore } from 'pinia';
import { api } from '@/api/client';
import type { Sequence } from '@/api/types';

export const useUploadStore = defineStore('upload', {
  state: () => ({
    sequences: [] as Sequence[],
    warnings: [] as string[],
    activeIndex: 0,
    loading: false,
    error: null as string | null,
  }),
  actions: {
    async ingest(file: File) {
      this.loading = true;
      this.error = null;
      try {
        const { sequences, warnings } = await api.uploadFile(file);
        this.sequences = sequences;
        this.warnings = warnings;
        this.activeIndex = 0;
      } catch (err) {
        this.error = (err as Error).message;
        this.sequences = [];
      } finally {
        this.loading = false;
      }
    },
    select(index: number) {
      if (index >= 0 && index < this.sequences.length) {
        this.activeIndex = index;
      }
    },
    clear() {
      this.sequences = [];
      this.warnings = [];
      this.activeIndex = 0;
    },
  },
});
