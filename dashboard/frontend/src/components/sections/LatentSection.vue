<script setup lang="ts">
import { onMounted, ref } from 'vue';
import LatentScatter from '@/components/charts/LatentScatter.vue';
import { api } from '@/api/client';
import { useSampleStore } from '@/stores/sample';
import type { LatentPoint } from '@/api/types';

const points = ref<LatentPoint[] | null>(null);
const colorBy = ref<'truth' | 'pred_a1' | 'pred_a2' | 'agreement'>('truth');
const error = ref<string | null>(null);
const sample = useSampleStore();

onMounted(async () => {
  try {
    const response = await api.latent();
    points.value = response.points;
  } catch (err) {
    error.value = (err as Error).message;
  }
});

async function openSample(id: number) {
  const compare = await api.latentSample(id);
  await sample.loadPrecomputed(compare);
}
</script>

<template>
  <section class="space-y-6 p-6">
    <div class="card p-5">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p class="section-title">UMAP projection of pooled hidden states</p>
          <p class="mt-1 text-xs text-slate-400">
            Click any point to load that sample into the focused view.
          </p>
        </div>
        <div class="flex gap-2">
          <button
            v-for="opt in [
              { id: 'truth', label: 'Ground truth' },
              { id: 'pred_a1', label: 'A1 prediction' },
              { id: 'pred_a2', label: 'A2 prediction' },
              { id: 'agreement', label: 'Agreement' },
            ] as const"
            :key="opt.id"
            :class="[
              'rounded-lg border px-3 py-1.5 text-xs transition',
              colorBy === opt.id
                ? 'border-cyan-400 bg-cyan-500/10 text-cyan-100'
                : 'border-slate-700 text-slate-300 hover:border-cyan-400',
            ]"
            @click="colorBy = opt.id"
          >
            {{ opt.label }}
          </button>
        </div>
      </div>
      <p v-if="error" class="mt-3 text-xs text-rose-300">{{ error }}</p>
      <LatentScatter v-if="points" :points="points" :color-by="colorBy" @select="openSample" />
      <p v-else-if="!error" class="mt-6 text-xs text-slate-500">Loading latent space…</p>
    </div>
  </section>
</template>
