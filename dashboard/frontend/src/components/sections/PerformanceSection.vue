<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { usePlotly } from '@/composables/usePlotly';
import type { Data, Layout } from 'plotly.js-dist-min';
import { api } from '@/api/client';
import type { MetricsResponse } from '@/api/types';

const metrics = ref<MetricsResponse | null>(null);

onMounted(async () => {
  metrics.value = await api.metrics();
});

const lossTraces = computed<Data[]>(() => {
  const series: Data[] = [];
  const attention = metrics.value?.attention as { train_losses?: number[] } | null;
  const seq = metrics.value?.seq2seq as { ae_losses?: number[]; clf_losses?: number[] } | null;
  if (attention?.train_losses) {
    series.push({
      x: attention.train_losses.map((_, i) => i + 1),
      y: attention.train_losses,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'A1 · attention LSTM',
      line: { color: '#22d3ee', width: 2 },
      marker: { size: 6 },
    });
  }
  if (seq?.ae_losses) {
    series.push({
      x: seq.ae_losses.map((_, i) => i + 1),
      y: seq.ae_losses,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'A2 · autoencoder',
      line: { color: '#a855f7', width: 2 },
      marker: { size: 6 },
    });
  }
  if (seq?.clf_losses) {
    series.push({
      x: seq.clf_losses.map((_, i) => i + 1),
      y: seq.clf_losses,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'A2 · classifier',
      line: { color: '#f472b6', width: 2 },
      marker: { size: 6 },
    });
  }
  return series;
});

const lossLayout = computed<Partial<Layout>>(() => ({
  height: 320,
  xaxis: { title: { text: 'Epoch' } },
  yaxis: { title: { text: 'Loss' } },
}));

const { container } = usePlotly(lossTraces, lossLayout);
</script>

<template>
  <section class="space-y-6 p-6">
    <div class="card p-5">
      <p class="section-title">Training loss curves</p>
      <p class="mt-1 text-xs text-slate-400">
        Loaded from <code class="font-mono text-slate-200">models/metrics.json</code> as written by the train CLI.
      </p>
      <div v-if="lossTraces.length" ref="container" class="mt-3 w-full" />
      <p v-else class="mt-6 text-xs text-slate-500">
        No metrics file yet. Run <code class="font-mono">python -m rnn_defect_detection train ...</code> to populate.
      </p>
    </div>

    <div v-if="metrics?.attention" class="card p-5">
      <p class="section-title">Approach 1 · per-class metrics</p>
      <pre class="mt-3 overflow-auto rounded-md border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-200">{{
        JSON.stringify(metrics.attention.per_class_metrics, null, 2)
      }}</pre>
    </div>

    <div v-if="metrics?.seq2seq" class="card p-5">
      <p class="section-title">Approach 2 · per-class metrics</p>
      <pre class="mt-3 overflow-auto rounded-md border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-200">{{
        JSON.stringify(metrics.seq2seq.per_class_metrics, null, 2)
      }}</pre>
    </div>
  </section>
</template>
