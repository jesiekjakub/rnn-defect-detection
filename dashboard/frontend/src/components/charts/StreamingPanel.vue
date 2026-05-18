<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { Play, Pause, RotateCw } from 'lucide-vue-next';
import { DEFECT_COLORS, DEFECT_NAMES } from '@/api/types';
import type { StreamSnapshot } from '@/api/types';

const props = defineProps<{
  snapshots: StreamSnapshot[];
}>();

const cursor = ref(0);
const playing = ref(false);
let timer: number | null = null;

const lastT = computed(() => props.snapshots[props.snapshots.length - 1]?.t ?? 0);
const currentSnapshot = computed(() => props.snapshots[cursor.value] ?? props.snapshots[0]);

const traces = computed<Data[]>(() => {
  if (props.snapshots.length === 0) return [];
  const cursorT = currentSnapshot.value?.t ?? 0;
  const visible = props.snapshots.slice(0, cursor.value + 1);
  const series: Data[] = [];
  for (let cls = 0; cls < DEFECT_NAMES.length; cls += 1) {
    series.push({
      x: visible.map((s) => s.t),
      y: visible.map((s) => s.probs_a1[cls]),
      type: 'scatter',
      mode: 'lines',
      name: `A1 · ${DEFECT_NAMES[cls]}`,
      line: { color: DEFECT_COLORS[cls], width: 2, dash: 'solid' },
      hovertemplate: `A1 ${DEFECT_NAMES[cls]} · t=%{x}<br>p=%{y:.3f}<extra></extra>`,
      legendgroup: `cls-${cls}`,
    });
    series.push({
      x: visible.map((s) => s.t),
      y: visible.map((s) => s.probs_a2[cls]),
      type: 'scatter',
      mode: 'lines',
      name: `A2 · ${DEFECT_NAMES[cls]}`,
      line: { color: DEFECT_COLORS[cls], width: 2, dash: 'dot' },
      hovertemplate: `A2 ${DEFECT_NAMES[cls]} · t=%{x}<br>p=%{y:.3f}<extra></extra>`,
      legendgroup: `cls-${cls}`,
      showlegend: false,
    });
  }
  // Cursor reference line.
  series.push({
    x: [cursorT, cursorT],
    y: [0, 1],
    type: 'scatter',
    mode: 'lines',
    line: { color: 'rgba(34,211,238,0.7)', width: 1.5, dash: 'dot' },
    showlegend: false,
    hoverinfo: 'skip',
  });
  return series;
});

const layout = computed<Partial<Layout>>(() => ({
  height: 340,
  yaxis: { range: [0, 1.05], title: { text: 'P(defect)' } },
  xaxis: { range: [props.snapshots[0]?.t ?? 0, lastT.value], title: { text: 'Timestep observed' } },
}));

const { container } = usePlotly(traces, layout);

function step() {
  cursor.value = Math.min(cursor.value + 1, props.snapshots.length - 1);
  if (cursor.value >= props.snapshots.length - 1) {
    playing.value = false;
  }
}

function togglePlay() {
  playing.value = !playing.value;
  if (playing.value) {
    if (cursor.value >= props.snapshots.length - 1) cursor.value = 0;
    timer = window.setInterval(step, 220);
  } else if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
}

function reset() {
  cursor.value = 0;
  playing.value = false;
  if (timer !== null) {
    window.clearInterval(timer);
    timer = null;
  }
}

watch(() => props.snapshots, reset);

onBeforeUnmount(() => {
  if (timer !== null) window.clearInterval(timer);
});
</script>

<template>
  <div class="space-y-3">
    <div ref="container" class="w-full" />
    <div class="flex items-center gap-3">
      <button
        class="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-slate-100 transition hover:border-cyan-400 hover:text-cyan-300"
        :aria-label="playing ? 'Pause' : 'Play'"
        @click="togglePlay"
      >
        <Play v-if="!playing" class="h-4 w-4" />
        <Pause v-else class="h-4 w-4" />
      </button>
      <button
        class="flex h-9 w-9 items-center justify-center rounded-lg border border-slate-700 bg-slate-900 text-slate-100 transition hover:border-cyan-400 hover:text-cyan-300"
        aria-label="Reset"
        @click="reset"
      >
        <RotateCw class="h-4 w-4" />
      </button>
      <input
        v-model.number="cursor"
        type="range"
        :min="0"
        :max="snapshots.length - 1"
        class="flex-1 accent-cyan-400"
      />
      <span class="tabular text-xs text-slate-400">t = {{ currentSnapshot?.t ?? 0 }}</span>
    </div>
  </div>
</template>
