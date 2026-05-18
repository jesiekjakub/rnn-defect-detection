<script setup lang="ts">
import { computed, onMounted, watch } from 'vue';
import Plotly from 'plotly.js-dist-min';
import type { Data, Layout } from 'plotly.js-dist-min';
import { usePlotly } from '@/composables/usePlotly';
import { DEFECT_COLORS, DEFECT_NAMES, SENSOR_COLORS } from '@/api/types';
import type { Sequence } from '@/api/types';

const props = defineProps<{
  sequence: Sequence;
  attention?: number[][];
  detectedDefects?: number[];
  pinnedTimestep?: number | null;
}>();

const emit = defineEmits<{
  (e: 'pin', t: number | null): void;
}>();

const traces = computed<Data[]>(() => {
  const t = props.sequence.x.map((_, i) => i);
  const sensors: Data[] = [];
  for (let s = 0; s < 3; s += 1) {
    sensors.push({
      x: t,
      y: props.sequence.x.map((row) => row[s]),
      type: 'scatter',
      mode: 'lines',
      name: `Sensor ${s}`,
      line: { color: SENSOR_COLORS[s], width: 2 },
      hovertemplate: `Sensor ${s}: %{y:.3f}<extra>t=%{x}</extra>`,
    });
  }

  const overlays: Data[] = [];
  if (props.attention && props.detectedDefects?.length) {
    for (const idx of props.detectedDefects) {
      const weights = props.attention.map((row) => row[idx]);
      const maxWeight = Math.max(...weights);
      if (maxWeight === 0) continue;
      const fill = weights.map((w) => (w > 0.05 ? 1 : 0));
      overlays.push({
        x: t,
        y: fill,
        yaxis: 'y2',
        type: 'scatter',
        mode: 'lines',
        name: `Attn: ${DEFECT_NAMES[idx]}`,
        fill: 'tozeroy',
        fillcolor: hexWithAlpha(DEFECT_COLORS[idx], 0.18),
        line: { color: DEFECT_COLORS[idx], width: 0 },
        hoverinfo: 'skip',
        showlegend: true,
      });
    }
  }

  return [...overlays, ...sensors];
});

const layout = computed<Partial<Layout>>(() => ({
  xaxis: { title: { text: 'Timestep', standoff: 6 } },
  yaxis: { title: { text: 'Signal value', standoff: 6 } },
  yaxis2: {
    overlaying: 'y',
    range: [0, 1],
    showgrid: false,
    showticklabels: false,
    zeroline: false,
  },
  shapes: props.pinnedTimestep !== null && props.pinnedTimestep !== undefined
    ? [
        {
          type: 'line',
          x0: props.pinnedTimestep,
          x1: props.pinnedTimestep,
          y0: 0,
          y1: 1,
          xref: 'x',
          yref: 'paper',
          line: { color: 'rgba(34,211,238,0.9)', width: 1.5, dash: 'dot' },
        },
      ]
    : [],
  height: 320,
}));

const { container } = usePlotly(traces, layout);

function hexWithAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

onMounted(() => {
  if (!container.value) return;
  (container.value as any).on('plotly_click', (event: any) => {
    const point = event.points?.[0];
    if (point) emit('pin', point.x as number);
  });
});

watch(() => props.pinnedTimestep, () => {
  if (container.value) Plotly.relayout(container.value, layout.value);
});
</script>

<template>
  <div ref="container" class="w-full" />
</template>
