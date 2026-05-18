<script setup lang="ts">
import { computed } from 'vue';
import SampleDesigner from '@/components/designer/SampleDesigner.vue';
import SensorSignalChart from '@/components/charts/SensorSignalChart.vue';
import ProbabilityBars from '@/components/charts/ProbabilityBars.vue';
import { useSampleStore } from '@/stores/sample';
import { useUiStore } from '@/stores/ui';

const sample = useSampleStore();
const ui = useUiStore();

const detected = computed(() => {
  const probs = sample.compare?.attention.probs ?? [];
  return probs.map((p, i) => (p > 0.5 ? i : -1)).filter((i) => i >= 0);
});
</script>

<template>
  <section class="grid grid-cols-1 gap-6 p-6 xl:grid-cols-[360px_minmax(0,1fr)]">
    <SampleDesigner />

    <div class="space-y-6">
      <div class="card p-5">
        <div class="flex items-center justify-between">
          <p class="section-title">Sensor signal</p>
          <p v-if="sample.sequence" class="tabular text-xs text-slate-400">
            T = {{ sample.sequence.x.length }} steps · origin {{ sample.sequence.origin }}
          </p>
        </div>
        <div v-if="sample.sequence" class="mt-3">
          <SensorSignalChart
            :sequence="sample.sequence"
            :attention="sample.compare?.attention.attention"
            :detected-defects="detected"
            :pinned-timestep="ui.pinnedTimestep"
            @pin="(t) => ui.pinTimestep(t)"
          />
        </div>
        <div v-else class="grid h-48 place-items-center text-sm text-slate-500">
          Generate a sample to see predictions.
        </div>
      </div>

      <div v-if="sample.compare" class="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div class="card p-5">
          <ProbabilityBars
            title="Approach 1 · attention LSTM"
            :probs="sample.compare.attention.probs"
            :truth="sample.sequence?.y_true"
          />
        </div>
        <div class="card p-5">
          <ProbabilityBars
            title="Approach 2 · seq2seq + classifier"
            :probs="sample.compare.seq2seq.probs"
            :truth="sample.sequence?.y_true"
          />
        </div>
      </div>
    </div>
  </section>
</template>
