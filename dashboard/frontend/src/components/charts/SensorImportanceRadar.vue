<script setup lang="ts">
import { computed } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { DEFECT_COLORS } from '@/api/types';
import type { AttentionExplanation } from '@/api/types';

const props = defineProps<{
  explanations: AttentionExplanation[];
}>();

const traces = computed<Data[]>(() => {
  const sensors = ['Sensor 0', 'Sensor 1', 'Sensor 2'];
  return props.explanations.map((exp) => ({
    type: 'scatterpolar',
    r: [...exp.sensor_importance, exp.sensor_importance[0]],
    theta: [...sensors, sensors[0]],
    fill: 'toself',
    name: exp.defect_name,
    line: { color: DEFECT_COLORS[exp.defect_index] },
    fillcolor: hexWithAlpha(DEFECT_COLORS[exp.defect_index], 0.18),
    hovertemplate: '%{theta}: %{r:.2%}<extra>%{fullData.name}</extra>',
  }));
});

const layout = computed<Partial<Layout>>(() => ({
  height: 280,
  polar: {
    bgcolor: 'rgba(0,0,0,0)',
    radialaxis: {
      visible: true,
      range: [0, 1],
      color: 'rgba(148,163,184,0.45)',
      gridcolor: 'rgba(148,163,184,0.18)',
      tickformat: '.0%',
    },
    angularaxis: {
      color: 'rgba(203,213,225,0.9)',
      gridcolor: 'rgba(148,163,184,0.18)',
    },
  },
  showlegend: true,
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
