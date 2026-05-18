<script setup lang="ts">
import { computed } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { DEFECT_NAMES } from '@/api/types';

const props = defineProps<{
  attention: number[][];
  pinnedTimestep?: number | null;
}>();

const traces = computed<Data[]>(() => {
  const seqLen = props.attention.length;
  const matrix: number[][] = [];
  for (let cls = 0; cls < DEFECT_NAMES.length; cls += 1) {
    const row: number[] = [];
    for (let t = 0; t < seqLen; t += 1) {
      row.push(props.attention[t][cls]);
    }
    matrix.push(row);
  }
  return [
    {
      z: matrix,
      x: Array.from({ length: seqLen }, (_, i) => i),
      y: [...DEFECT_NAMES],
      type: 'heatmap',
      colorscale: [
        [0, 'rgba(15,23,42,1)'],
        [0.2, 'rgba(34,211,238,0.25)'],
        [0.5, 'rgba(168,85,247,0.65)'],
        [1, 'rgba(244,114,182,1)'],
      ],
      colorbar: {
        thickness: 8,
        tickfont: { size: 10, color: '#cbd5f5' },
        outlinewidth: 0,
      },
      hovertemplate: '%{y}<br>t=%{x}<br>attn=%{z:.3f}<extra></extra>',
    },
  ];
});

const layout = computed<Partial<Layout>>(() => ({
  margin: { l: 90, r: 16, t: 16, b: 32 },
  xaxis: { title: { text: 'Timestep', standoff: 4 } },
  yaxis: { tickfont: { size: 11 } },
  shapes: props.pinnedTimestep !== null && props.pinnedTimestep !== undefined
    ? [
        {
          type: 'line',
          x0: props.pinnedTimestep,
          x1: props.pinnedTimestep,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: { color: 'rgba(34,211,238,0.9)', width: 1.5, dash: 'dot' },
        },
      ]
    : [],
  height: 220,
}));

const { container } = usePlotly(traces, layout);
</script>

<template>
  <div ref="container" class="w-full" />
</template>
