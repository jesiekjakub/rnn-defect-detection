<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import Plotly, { type Data, type Layout } from 'plotly.js-dist-min';
import { mergeLayout } from '@/composables/usePlotly';
import { DEFECT_COLORS, DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';
import type { LatentPoint } from '@/api/types';

const props = defineProps<{
  points: LatentPoint[];
  colorBy: 'truth' | 'pred_a1' | 'pred_a2' | 'agreement';
}>();

const emit = defineEmits<{
  (e: 'select', sampleId: number): void;
}>();

const container = ref<HTMLDivElement | null>(null);

function selectClass(labels: number[]): number {
  const idx = labels.findIndex((v) => v === 1);
  return idx === -1 ? -1 : idx;
}

const traces = computed<Data[]>(() => {
  if (props.colorBy === 'agreement') {
    const agreeX: number[] = [], agreeY: number[] = [], agreeIds: number[] = [];
    const disagreeX: number[] = [], disagreeY: number[] = [], disagreeIds: number[] = [];
    for (const p of props.points) {
      if (p.agreement) {
        agreeX.push(p.x); agreeY.push(p.y); agreeIds.push(p.sample_id);
      } else {
        disagreeX.push(p.x); disagreeY.push(p.y); disagreeIds.push(p.sample_id);
      }
    }
    return [
      {
        x: agreeX, y: agreeY,
        type: 'scattergl', mode: 'markers',
        name: 'A1 == A2',
        marker: { color: 'rgba(34,211,238,0.6)', size: 6 },
        customdata: agreeIds,
        hovertemplate: 'sample #%{customdata}<extra>agreement</extra>',
      },
      {
        x: disagreeX, y: disagreeY,
        type: 'scattergl', mode: 'markers',
        name: 'A1 ≠ A2',
        marker: { color: 'rgba(244,114,182,0.85)', size: 7, line: { color: 'white', width: 0.5 } },
        customdata: disagreeIds,
        hovertemplate: 'sample #%{customdata}<extra>disagreement</extra>',
      },
    ];
  }

  const buckets: { x: number[]; y: number[]; ids: number[] }[] = Array.from(
    { length: NUM_DEFECT_TYPES + 1 },
    () => ({ x: [], y: [], ids: [] }),
  );
  for (const point of props.points) {
    let cls = -1;
    if (props.colorBy === 'truth') cls = selectClass(point.y_true);
    if (props.colorBy === 'pred_a1') cls = selectClass(point.y_pred_a1);
    if (props.colorBy === 'pred_a2') cls = selectClass(point.y_pred_a2);
    const bucket = cls === -1 ? NUM_DEFECT_TYPES : cls;
    buckets[bucket].x.push(point.x);
    buckets[bucket].y.push(point.y);
    buckets[bucket].ids.push(point.sample_id);
  }
  const traces: Data[] = [];
  for (let i = 0; i < NUM_DEFECT_TYPES; i += 1) {
    if (buckets[i].x.length === 0) continue;
    traces.push({
      x: buckets[i].x,
      y: buckets[i].y,
      type: 'scattergl',
      mode: 'markers',
      name: DEFECT_NAMES[i],
      marker: { color: DEFECT_COLORS[i], size: 6, opacity: 0.85 },
      customdata: buckets[i].ids,
      hovertemplate: `${DEFECT_NAMES[i]}<br>sample #%{customdata}<extra></extra>`,
    });
  }
  if (buckets[NUM_DEFECT_TYPES].x.length) {
    traces.push({
      x: buckets[NUM_DEFECT_TYPES].x,
      y: buckets[NUM_DEFECT_TYPES].y,
      type: 'scattergl',
      mode: 'markers',
      name: 'Healthy / multi-label',
      marker: { color: 'rgba(148,163,184,0.45)', size: 5 },
      customdata: buckets[NUM_DEFECT_TYPES].ids,
      hovertemplate: 'sample #%{customdata}<extra>healthy / multi</extra>',
    });
  }
  return traces;
});

const layout = computed<Partial<Layout>>(() => ({
  height: 500,
  xaxis: { title: { text: 'UMAP 1' }, zeroline: false },
  yaxis: { title: { text: 'UMAP 2' }, zeroline: false },
  legend: { orientation: 'v', x: 1.02, y: 1, font: { size: 11 } },
  margin: { l: 48, r: 200, t: 24, b: 48 },
}));

async function render() {
  if (!container.value) return;
  await Plotly.react(container.value, traces.value, mergeLayout(layout.value), {
    displaylogo: false,
    responsive: true,
  });
}

onMounted(() => {
  render();
  if (!container.value) return;
  (container.value as any).on('plotly_click', (event: any) => {
    const point = event.points?.[0];
    const id = point?.customdata;
    if (typeof id === 'number') emit('select', id);
  });
});

watch([traces, layout], render, { deep: true });
</script>

<template>
  <div ref="container" class="w-full" />
</template>
