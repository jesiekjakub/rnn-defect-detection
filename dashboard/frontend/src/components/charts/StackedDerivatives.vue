<script setup lang="ts">
import { computed } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { DEFECT_COLORS, SENSOR_COLORS } from '@/api/types';
import type { VerifiedRegion } from '@/api/types';

const props = defineProps<{
  original: number[][];
  residual: number[][];
  velocity: number[][];
  acceptedRegions?: VerifiedRegion[];
}>();

const traces = computed<Data[]>(() => {
  const t = props.original.map((_, i) => i);
  const traces: Data[] = [];
  const groups = [
    { y: props.original, name: 'Sensor', subplot: 'y' },
    { y: props.residual, name: 'Residual', subplot: 'y2' },
    { y: props.velocity, name: 'Velocity', subplot: 'y3' },
  ];
  for (const group of groups) {
    for (let s = 0; s < 3; s += 1) {
      traces.push({
        x: t,
        y: group.y.map((row) => row[s]),
        type: 'scatter',
        mode: 'lines',
        name: `${group.name} S${s}`,
        legendgroup: `s${s}`,
        showlegend: group.subplot === 'y',
        line: { color: SENSOR_COLORS[s], width: 1.5 },
        yaxis: group.subplot,
        hovertemplate: `${group.name} S${s}: %{y:.3f}<extra>t=%{x}</extra>`,
      });
    }
  }
  return traces;
});

const layout = computed<Partial<Layout>>(() => ({
  grid: { rows: 3, columns: 1, pattern: 'independent' },
  height: 460,
  yaxis: { domain: [0.7, 1], title: { text: 'Signal', standoff: 4 } },
  yaxis2: { domain: [0.36, 0.66], title: { text: 'Residual', standoff: 4 } },
  yaxis3: { domain: [0.0, 0.32], title: { text: 'Velocity', standoff: 4 } },
  xaxis: { matches: 'x', title: { text: 'Timestep', standoff: 4 } },
  shapes: (props.acceptedRegions ?? []).flatMap((region) => {
    const color = DEFECT_COLORS[region.defect_index];
    const fill = hexWithAlpha(color, 0.14);
    return ['y', 'y2', 'y3'].map((axis) => ({
      type: 'rect' as const,
      xref: 'x' as const,
      yref: axis as 'y',
      x0: region.start,
      x1: region.end,
      y0: 0,
      y1: 1,
      fillcolor: fill,
      line: { width: 0 },
      layer: 'below' as const,
    }));
  }),
}));

const { container } = usePlotly(traces, layout);

function hexWithAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
</script>

<template>
  <div ref="container" class="w-full" />
</template>
