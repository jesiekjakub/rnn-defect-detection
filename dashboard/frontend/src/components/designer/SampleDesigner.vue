<script setup lang="ts">
import { computed } from 'vue';
import { Dice5, Wand2 } from 'lucide-vue-next';
import { useSampleStore } from '@/stores/sample';
import { DEFECT_COLORS, DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';

const store = useSampleStore();

const defects = computed(() =>
  Array.from({ length: NUM_DEFECT_TYPES }, (_, i) => ({
    index: i,
    name: DEFECT_NAMES[i],
    color: DEFECT_COLORS[i],
    enabled: store.spec.defects[i],
  })),
);

function setSeed(value: string) {
  const trimmed = value.trim();
  store.setSpec({ seed: trimmed === '' ? null : Number(trimmed) });
}
</script>

<template>
  <div class="card flex flex-col gap-5 p-5">
    <div class="flex items-center justify-between">
      <div>
        <p class="section-title">Sample designer</p>
        <p class="mt-1 text-lg font-semibold text-slate-100">Inject defects, set noise, generate</p>
      </div>
      <button
        class="flex h-9 items-center gap-2 rounded-lg border border-slate-700 px-3 text-xs text-slate-300 transition hover:border-cyan-400 hover:text-cyan-300"
        @click="store.surpriseMe()"
      >
        <Dice5 class="h-4 w-4" />
        Surprise me
      </button>
    </div>

    <div class="grid grid-cols-1 gap-2">
      <button
        v-for="defect in defects"
        :key="defect.index"
        :class="[
          'flex items-center justify-between rounded-lg border px-3 py-2 text-left transition',
          defect.enabled
            ? 'border-cyan-400/60 bg-cyan-500/10'
            : 'border-slate-800 bg-slate-900/40 hover:border-slate-700',
        ]"
        @click="store.toggleDefect(defect.index)"
      >
        <span class="flex items-center gap-3 text-sm font-medium text-slate-100">
          <span class="h-2.5 w-2.5 rounded-full" :style="{ background: defect.color }" />
          {{ defect.name }}
        </span>
        <span class="text-[10px] uppercase tracking-wider" :class="defect.enabled ? 'text-cyan-200' : 'text-slate-500'">
          {{ defect.enabled ? 'inject' : 'off' }}
        </span>
      </button>
    </div>

    <div class="grid grid-cols-2 gap-4">
      <label class="flex flex-col gap-1 text-xs text-slate-400">
        <span>Sequence length</span>
        <input
          v-model.number="store.spec.seq_len"
          type="range"
          min="20"
          max="120"
          step="1"
          class="accent-cyan-400"
        />
        <span class="tabular text-slate-300">{{ store.spec.seq_len }} steps</span>
      </label>
      <label class="flex flex-col gap-1 text-xs text-slate-400">
        <span>Noise scale</span>
        <input
          v-model.number="store.spec.noise_scale"
          type="range"
          min="0"
          max="1"
          step="0.05"
          class="accent-cyan-400"
        />
        <span class="tabular text-slate-300">{{ store.spec.noise_scale.toFixed(2) }}</span>
      </label>
      <label class="col-span-2 flex flex-col gap-1 text-xs text-slate-400">
        <span>Seed (blank = random)</span>
        <input
          :value="store.spec.seed ?? ''"
          type="text"
          inputmode="numeric"
          placeholder="42"
          class="rounded-md border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm text-slate-100 focus:border-cyan-400 focus:outline-none"
          @input="setSeed(($event.target as HTMLInputElement).value)"
        />
      </label>
    </div>

    <button
      class="flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-cyan-500 to-fuchsia-500 px-4 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/20 transition hover:from-cyan-400 hover:to-fuchsia-400 disabled:opacity-60"
      :disabled="store.loading"
      @click="store.generateAndPredict()"
    >
      <Wand2 class="h-4 w-4" />
      {{ store.loading ? 'Generating…' : 'Generate & predict' }}
    </button>

    <p v-if="store.error" class="text-xs text-rose-300">{{ store.error }}</p>
  </div>
</template>
