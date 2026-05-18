<script setup lang="ts">
import { computed, watch } from 'vue';
import UploadDropzone from '@/components/upload/UploadDropzone.vue';
import SensorSignalChart from '@/components/charts/SensorSignalChart.vue';
import ProbabilityBars from '@/components/charts/ProbabilityBars.vue';
import { useUploadStore } from '@/stores/upload';
import { useSampleStore } from '@/stores/sample';
import { useUiStore } from '@/stores/ui';

const upload = useUploadStore();
const sample = useSampleStore();
const ui = useUiStore();

const active = computed(() => upload.sequences[upload.activeIndex] ?? null);

watch(active, (next) => {
  if (next) void sample.loadSequence(next);
});

const detected = computed(() => {
  const probs = sample.compare?.attention.probs ?? [];
  return probs.map((p, i) => (p > 0.5 ? i : -1)).filter((i) => i >= 0);
});
</script>

<template>
  <section class="grid grid-cols-1 gap-6 p-6 xl:grid-cols-[360px_minmax(0,1fr)]">
    <div class="space-y-4">
      <UploadDropzone />
      <div v-if="upload.sequences.length" class="card max-h-96 overflow-y-auto scrollbar-thin p-3">
        <p class="section-title px-2 pb-2">{{ upload.sequences.length }} sequences</p>
        <button
          v-for="(seq, i) in upload.sequences"
          :key="seq.sequence_id ?? i"
          :class="[
            'flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-xs transition',
            i === upload.activeIndex
              ? 'bg-cyan-500/10 text-cyan-100 ring-1 ring-cyan-500/30'
              : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-100',
          ]"
          @click="upload.select(i)"
        >
          <span class="truncate font-medium">{{ seq.sequence_id ?? `seq_${i}` }}</span>
          <span class="tabular text-[10px] text-slate-500">{{ seq.x.length }} steps</span>
        </button>
      </div>
    </div>

    <div class="space-y-6">
      <div class="card p-5">
        <p class="section-title">Sensor signal</p>
        <div v-if="sample.sequence && sample.sequence.origin === 'upload'" class="mt-3">
          <SensorSignalChart
            :sequence="sample.sequence"
            :attention="sample.compare?.attention.attention"
            :detected-defects="detected"
            :pinned-timestep="ui.pinnedTimestep"
            @pin="(t) => ui.pinTimestep(t)"
          />
        </div>
        <div v-else class="grid h-48 place-items-center text-sm text-slate-500">
          Drop a file or pick one from the dropzone to begin.
        </div>
      </div>

      <div v-if="sample.compare && sample.sequence?.origin === 'upload'" class="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div class="card p-5">
          <ProbabilityBars title="Approach 1" :probs="sample.compare.attention.probs" :truth="sample.sequence.y_true" />
        </div>
        <div class="card p-5">
          <ProbabilityBars title="Approach 2" :probs="sample.compare.seq2seq.probs" :truth="sample.sequence.y_true" />
        </div>
      </div>
    </div>
  </section>
</template>
