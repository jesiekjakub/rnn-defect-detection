<script setup lang="ts">
import { computed } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';
import type { BatchRow } from '@/api/types';

const props = defineProps<{
  rows: BatchRow[];
  approach: 'attention' | 'seq2seq';
}>();

const traces = computed<Data[]>(() => {
  const matrix: number[][] = Array.from({ length: NUM_DEFECT_TYPES }, () =>
    Array.from({ length: NUM_DEFECT_TYPES }, () => 0),
  );
  for (const row of props.rows) {
    const pred = props.approach === 'attention' ? row.pred_a1 : row.pred_a2;
    for (let i = 0; i < NUM_DEFECT_TYPES; i += 1) {
      if (!pred[i]) continue;
      for (let j = 0; j < NUM_DEFECT_TYPES; j += 1) {
        if (pred[j]) matrix[i][j] += 1;
      }
    }
  }
  for (let i = 0; i < NUM_DEFECT_TYPES; i += 1) {
    const diag = matrix[i][i] || 1;
    for (let j = 0; j < NUM_DEFECT_TYPES; j += 1) {
      matrix[i][j] = matrix[i][j] / diag;
    }
  }
  return [
    {
      z: matrix,
      x: [...DEFECT_NAMES],
      y: [...DEFECT_NAMES],
      type: 'heatmap',
      colorscale: [
        [0, 'rgba(15,23,42,1)'],
        [0.4, 'rgba(34,211,238,0.4)'],
        [1, 'rgba(244,114,182,0.95)'],
      ],
      hovertemplate: 'P(%{y} | %{x}) = %{z:.2f}<extra></extra>',
      colorbar: { thickness: 8, tickformat: '.0%', outlinewidth: 0 },
    },
  ];
});

const layout = computed<Partial<Layout>>(() => ({
  height: 320,
  margin: { l: 120, r: 16, t: 16, b: 80 },
  xaxis: { tickangle: -25 },
}));

const { container } = usePlotly(traces, layout);
</script>

<template>
  <div ref="container" class="w-full" />
</template>
