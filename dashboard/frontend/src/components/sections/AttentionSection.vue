<script setup lang="ts">
import { computed } from 'vue';
import SensorSignalChart from '@/components/charts/SensorSignalChart.vue';
import AttentionHeatmap from '@/components/charts/AttentionHeatmap.vue';
import SensorImportanceRadar from '@/components/charts/SensorImportanceRadar.vue';
import { useSampleStore } from '@/stores/sample';
import { useUiStore } from '@/stores/ui';
import { DEFECT_COLORS } from '@/api/types';

const sample = useSampleStore();
const ui = useUiStore();

const detected = computed(() => sample.compare?.attention.probs.map((p, i) => (p > 0.5 ? i : -1)).filter((i) => i >= 0) ?? []);
const pinnedDetail = computed(() => {
  if (sample.compare === null || ui.pinnedTimestep === null) return null;
  const row = sample.compare.attention.attention[ui.pinnedTimestep];
  if (!row) return null;
  return row.map((value, cls) => ({ cls, value }));
});
</script>

<template>
  <section v-if="sample.compare && sample.sequence" class="space-y-6 p-6">
    <div class="card p-5">
      <p class="section-title">Attention overlay</p>
      <SensorSignalChart
        :sequence="sample.sequence"
        :attention="sample.compare.attention.attention"
        :detected-defects="detected"
        :pinned-timestep="ui.pinnedTimestep"
        @pin="(t) => ui.pinTimestep(t)"
      />
    </div>

    <div class="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <div class="card p-5">
        <p class="section-title">Per-class attention heatmap</p>
        <AttentionHeatmap
          :attention="sample.compare.attention.attention"
          :pinned-timestep="ui.pinnedTimestep"
        />
      </div>
      <div class="card p-5">
        <p class="section-title">Sensor importance (variance ratio)</p>
        <SensorImportanceRadar v-if="sample.compare.attention.explanations.length"
          :explanations="sample.compare.attention.explanations"
        />
        <p v-else class="mt-6 text-xs text-slate-500">No defect detected in this sample.</p>
      </div>
    </div>

    <div v-if="ui.pinnedTimestep !== null && pinnedDetail" class="card p-5">
      <p class="section-title">Pinned timestep · t = {{ ui.pinnedTimestep }}</p>
      <div class="mt-3 grid grid-cols-2 gap-3 md:grid-cols-5">
        <div
          v-for="entry in pinnedDetail"
          :key="entry.cls"
          class="rounded-lg border border-slate-800 bg-slate-950/40 p-3"
        >
          <p class="text-[10px] uppercase tracking-wider text-slate-400">class {{ entry.cls }}</p>
          <p class="mt-1 tabular text-lg font-semibold" :style="{ color: DEFECT_COLORS[entry.cls] }">
            {{ (entry.value * 100).toFixed(1) }}%
          </p>
        </div>
      </div>
    </div>

    <div v-if="sample.compare.attention.explanations.length" class="card p-5">
      <p class="section-title">Root cause summary</p>
      <ul class="mt-3 space-y-2 text-sm">
        <li
          v-for="exp in sample.compare.attention.explanations"
          :key="exp.defect_index"
          class="flex items-start gap-3 rounded-lg bg-slate-950/40 p-3"
        >
          <span
            class="mt-1 h-2.5 w-2.5 rounded-full"
            :style="{ background: DEFECT_COLORS[exp.defect_index] }"
          />
          <div class="flex-1">
            <p class="text-slate-100">
              {{ exp.defect_name }}
              <span class="ml-2 tabular text-xs text-slate-400">conf {{ (exp.confidence * 100).toFixed(1) }}%</span>
            </p>
            <p class="mt-1 text-xs text-slate-400">
              Ranges {{ exp.ranges.map(([a, b]) => `${a}–${b}`).join(', ') }} · top sensor
              {{ exp.sensor_importance.indexOf(Math.max(...exp.sensor_importance)) }}
            </p>
          </div>
        </li>
      </ul>
    </div>
  </section>
  <section v-else class="grid place-items-center p-12 text-sm text-slate-500">
    Run a sample on the Live Demo tab first.
  </section>
</template>
