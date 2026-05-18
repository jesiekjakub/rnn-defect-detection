<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useThresholdStore } from '@/stores/threshold';
import ConfusionMatrix from '@/components/charts/ConfusionMatrix.vue';
import RocPrCurves from '@/components/charts/RocPrCurves.vue';
import { DEFECT_COLORS, DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';

const threshold = useThresholdStore();
const curveMode = ref<'roc' | 'pr'>('roc');

onMounted(async () => {
  if (!threshold.metrics) await threshold.refreshMetrics();
  if (!threshold.curves) await threshold.refreshCurves();
});

function onSlide(i: number, event: Event) {
  const value = Number((event.target as HTMLInputElement).value);
  void threshold.setThreshold(i, value);
}
</script>

<template>
  <section class="space-y-6 p-6">
    <div class="card p-5">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <p class="section-title">Approach</p>
        <div class="flex gap-2">
          <button
            v-for="opt in ['attention', 'seq2seq'] as const"
            :key="opt"
            :class="[
              'rounded-lg border px-3 py-1.5 text-xs transition',
              threshold.approach === opt
                ? 'border-cyan-400 bg-cyan-500/10 text-cyan-100'
                : 'border-slate-700 text-slate-300 hover:border-cyan-400',
            ]"
            @click="threshold.setApproach(opt)"
          >
            {{ opt }}
          </button>
        </div>
      </div>

      <div class="mt-5 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-5">
        <div
          v-for="i in NUM_DEFECT_TYPES"
          :key="i - 1"
          class="rounded-lg border border-slate-800 bg-slate-950/40 p-3"
        >
          <div class="flex items-center justify-between text-xs">
            <span class="flex items-center gap-2 text-slate-200">
              <span class="h-2 w-2 rounded-full" :style="{ background: DEFECT_COLORS[i - 1] }" />
              {{ DEFECT_NAMES[i - 1] }}
            </span>
            <span class="tabular text-slate-400">θ = {{ threshold.thresholds[i - 1].toFixed(2) }}</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            class="mt-3 w-full accent-cyan-400"
            :value="threshold.thresholds[i - 1]"
            @input="onSlide(i - 1, $event)"
          />
          <div v-if="threshold.metrics?.per_class[i - 1]" class="mt-3 grid grid-cols-3 gap-2 text-[11px]">
            <div>
              <p class="text-slate-500">P</p>
              <p class="tabular text-slate-100">{{ threshold.metrics.per_class[i - 1].precision.toFixed(3) }}</p>
            </div>
            <div>
              <p class="text-slate-500">R</p>
              <p class="tabular text-slate-100">{{ threshold.metrics.per_class[i - 1].recall.toFixed(3) }}</p>
            </div>
            <div>
              <p class="text-slate-500">F1</p>
              <p class="tabular text-slate-100">{{ threshold.metrics.per_class[i - 1].f1.toFixed(3) }}</p>
            </div>
          </div>
        </div>
      </div>

      <div v-if="threshold.metrics" class="mt-5 grid grid-cols-3 gap-3 text-center">
        <div class="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <p class="text-[10px] uppercase tracking-wider text-slate-400">macro precision</p>
          <p class="tabular text-lg text-slate-100">{{ threshold.metrics.macro_precision.toFixed(3) }}</p>
        </div>
        <div class="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <p class="text-[10px] uppercase tracking-wider text-slate-400">macro recall</p>
          <p class="tabular text-lg text-slate-100">{{ threshold.metrics.macro_recall.toFixed(3) }}</p>
        </div>
        <div class="rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <p class="text-[10px] uppercase tracking-wider text-slate-400">macro F1</p>
          <p class="tabular text-lg text-cyan-200">{{ threshold.metrics.macro_f1.toFixed(3) }}</p>
        </div>
      </div>
    </div>

    <div v-if="threshold.metrics" class="card p-5">
      <p class="section-title">Confusion matrices (per defect, current thresholds)</p>
      <ConfusionMatrix :per-class="threshold.metrics.per_class" />
    </div>

    <div v-if="threshold.curves" class="card p-5">
      <div class="flex items-center justify-between">
        <p class="section-title">{{ curveMode === 'roc' ? 'ROC curves' : 'Precision-recall curves' }}</p>
        <div class="flex gap-2">
          <button
            v-for="opt in ['roc', 'pr'] as const"
            :key="opt"
            :class="[
              'rounded-lg border px-2.5 py-1 text-[11px] transition',
              curveMode === opt
                ? 'border-cyan-400 bg-cyan-500/10 text-cyan-100'
                : 'border-slate-700 text-slate-300 hover:border-cyan-400',
            ]"
            @click="curveMode = opt"
          >
            {{ opt.toUpperCase() }}
          </button>
        </div>
      </div>
      <RocPrCurves :curves="threshold.curves" :thresholds="threshold.thresholds" :mode="curveMode" />
    </div>
  </section>
</template>
