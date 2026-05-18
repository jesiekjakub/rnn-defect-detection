<script setup lang="ts">
import { ref, watch } from 'vue';
import { api } from '@/api/client';
import StreamingPanel from '@/components/charts/StreamingPanel.vue';
import { useSampleStore } from '@/stores/sample';
import type { StreamSnapshot } from '@/api/types';

const sample = useSampleStore();
const snapshots = ref<StreamSnapshot[]>([]);
const windowSize = ref(12);
const stride = ref(1);
const loading = ref(false);
const error = ref<string | null>(null);

async function recompute() {
  if (!sample.sequence) return;
  loading.value = true;
  error.value = null;
  try {
    const response = await api.predictStream(sample.sequence, windowSize.value, stride.value);
    snapshots.value = response.snapshots;
  } catch (err) {
    error.value = (err as Error).message;
  } finally {
    loading.value = false;
  }
}

watch(() => [sample.sequence, windowSize.value, stride.value], recompute, { immediate: true });
</script>

<template>
  <section class="space-y-6 p-6">
    <div class="card p-5">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p class="section-title">Streaming replay</p>
          <p class="mt-1 text-xs text-slate-400">
            Each snapshot re-runs both models on the first <span class="tabular">t</span> timesteps.
          </p>
        </div>
        <div class="flex items-center gap-3 text-xs text-slate-300">
          <label class="flex items-center gap-2">
            window
            <input
              v-model.number="windowSize"
              type="number"
              min="4"
              max="60"
              class="w-16 rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-right text-xs"
            />
          </label>
          <label class="flex items-center gap-2">
            stride
            <input
              v-model.number="stride"
              type="number"
              min="1"
              max="5"
              class="w-16 rounded-md border border-slate-800 bg-slate-950/60 px-2 py-1 text-right text-xs"
            />
          </label>
        </div>
      </div>
      <p v-if="loading" class="mt-3 text-xs text-cyan-200">computing snapshots…</p>
      <p v-if="error" class="mt-3 text-xs text-rose-300">{{ error }}</p>
      <StreamingPanel v-if="snapshots.length" :snapshots="snapshots" />
    </div>
  </section>
</template>
