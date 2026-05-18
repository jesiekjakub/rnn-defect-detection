<script setup lang="ts">
import { computed } from 'vue';
import SensorSignalChart from '@/components/charts/SensorSignalChart.vue';
import ProbabilityBars from '@/components/charts/ProbabilityBars.vue';
import AgreementMatrix from '@/components/charts/AgreementMatrix.vue';
import { useSampleStore } from '@/stores/sample';
import { useUiStore } from '@/stores/ui';

const sample = useSampleStore();
const ui = useUiStore();

const predA1 = computed(() => sample.compare?.attention.probs.map((p) => p > 0.5) ?? []);
const predA2 = computed(() => sample.compare?.seq2seq.probs.map((p) => p > 0.5) ?? []);
const detected = computed(() => sample.compare?.attention.probs.map((p, i) => (p > 0.5 ? i : -1)).filter((i) => i >= 0) ?? []);
</script>

<template>
  <section v-if="sample.compare && sample.sequence" class="space-y-6 p-6">
    <div class="card p-5">
      <p class="section-title">Per-defect agreement</p>
      <AgreementMatrix
        :pred-a1="predA1"
        :pred-a2="predA2"
        :truth="sample.sequence.y_true"
      />
    </div>

    <div class="grid grid-cols-1 gap-6 xl:grid-cols-2">
      <div class="card p-5">
        <p class="section-title">Approach 1 · attention LSTM</p>
        <div class="mt-3">
          <SensorSignalChart
            :sequence="sample.sequence"
            :attention="sample.compare.attention.attention"
            :detected-defects="detected"
            :pinned-timestep="ui.pinnedTimestep"
            @pin="(t) => ui.pinTimestep(t)"
          />
        </div>
        <div class="mt-4">
          <ProbabilityBars :probs="sample.compare.attention.probs" :truth="sample.sequence.y_true" />
        </div>
      </div>

      <div class="card p-5">
        <p class="section-title">Approach 2 · seq2seq + classifier</p>
        <div class="mt-3">
          <SensorSignalChart
            :sequence="sample.sequence"
            :pinned-timestep="ui.pinnedTimestep"
            @pin="(t) => ui.pinTimestep(t)"
          />
        </div>
        <div class="mt-4 space-y-4">
          <ProbabilityBars :probs="sample.compare.seq2seq.probs" :truth="sample.sequence.y_true" />
          <ul v-if="sample.compare.seq2seq.accepted_regions.length" class="space-y-1 text-xs text-slate-300">
            <li
              v-for="region in sample.compare.seq2seq.accepted_regions"
              :key="`${region.defect_index}-${region.start}`"
              class="rounded-md border border-slate-800 px-2 py-1.5"
            >
              {{ region.defect_name }} · t {{ region.start }}–{{ region.end }} · p {{ (region.local_probability * 100).toFixed(1) }}%
            </li>
          </ul>
        </div>
      </div>
    </div>
  </section>
  <section v-else class="grid place-items-center p-12 text-sm text-slate-500">
    Run a sample on the Live Demo tab first.
  </section>
</template>
