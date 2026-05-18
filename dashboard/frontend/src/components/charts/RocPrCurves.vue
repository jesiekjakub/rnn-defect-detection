<script setup lang="ts">
import { computed } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { DEFECT_COLORS, DEFECT_NAMES } from '@/api/types';
import type { CurvePoint } from '@/api/types';

const props = defineProps<{
  curves: CurvePoint[][];
  thresholds: number[];
  mode?: 'roc' | 'pr';
}>();

const traces = computed<Data[]>(() => {
  const out: Data[] = [];
  const isRoc = props.mode !== 'pr';
  props.curves.forEach((curve, cls) => {
    const xs = curve.map((p) => (isRoc ? p.fpr : p.recall));
    const ys = curve.map((p) => (isRoc ? p.tpr : p.precision));
    out.push({
      x: xs,
      y: ys,
      type: 'scatter',
      mode: 'lines',
      name: DEFECT_NAMES[cls],
      line: { color: DEFECT_COLORS[cls], width: 2 },
      hovertemplate: `${DEFECT_NAMES[cls]} · ${isRoc ? 'FPR' : 'recall'}: %{x:.3f}<br>${isRoc ? 'TPR' : 'precision'}: %{y:.3f}<extra></extra>`,
    });
    // Marker at the active threshold for this class.
    const activeThr = props.thresholds[cls];
    const active = curve.reduce((best, point) =>
      Math.abs(point.threshold - activeThr) < Math.abs(best.threshold - activeThr) ? point : best,
    );
    out.push({
      x: [isRoc ? active.fpr : active.recall],
      y: [isRoc ? active.tpr : active.precision],
      type: 'scatter',
      mode: 'markers',
      name: `${DEFECT_NAMES[cls]} · θ=${activeThr.toFixed(2)}`,
      marker: { color: DEFECT_COLORS[cls], size: 10, line: { color: 'white', width: 1 } },
      showlegend: false,
      hoverinfo: 'skip',
    });
  });
  return out;
});

const layout = computed<Partial<Layout>>(() => ({
  height: 320,
  xaxis: {
    range: [0, 1],
    title: { text: props.mode === 'pr' ? 'Recall' : 'False positive rate' },
  },
  yaxis: {
    range: [0, 1.05],
    title: { text: props.mode === 'pr' ? 'Precision' : 'True positive rate' },
  },
  margin: { l: 56, r: 16, t: 16, b: 48 },
}));

const { container } = usePlotly(traces, layout);
</script>

<template>
  <div ref="container" class="w-full" />
</template>
