<script setup lang="ts">
import { computed } from 'vue';
import { DEFECT_COLORS, DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';

const props = defineProps<{
  probs: number[];
  truth?: number[] | null;
  title?: string;
  threshold?: number;
}>();

const rows = computed(() =>
  Array.from({ length: NUM_DEFECT_TYPES }, (_, i) => ({
    index: i,
    name: DEFECT_NAMES[i],
    color: DEFECT_COLORS[i],
    prob: props.probs?.[i] ?? 0,
    truth: props.truth?.[i] ?? null,
    above: (props.probs?.[i] ?? 0) > (props.threshold ?? 0.5),
  })),
);
</script>

<template>
  <div class="space-y-2">
    <p v-if="title" class="section-title">{{ title }}</p>
    <div v-for="row in rows" :key="row.index" class="space-y-1">
      <div class="flex items-center justify-between text-xs">
        <span class="flex items-center gap-2 text-slate-300">
          <span class="h-2 w-2 rounded-full" :style="{ background: row.color }" />
          {{ row.name }}
          <span
            v-if="row.truth !== null"
            class="rounded-full px-1.5 py-0.5 text-[10px] uppercase tracking-wide"
            :class="row.truth === 1 ? 'bg-emerald-500/15 text-emerald-300' : 'bg-slate-700/40 text-slate-400'"
          >
            truth {{ row.truth }}
          </span>
        </span>
        <span class="tabular text-slate-200">{{ (row.prob * 100).toFixed(1) }}%</span>
      </div>
      <div class="relative h-2 overflow-hidden rounded-full bg-slate-800/80">
        <div
          class="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500"
          :class="row.above ? 'opacity-100' : 'opacity-50'"
          :style="{ width: `${Math.min(100, row.prob * 100)}%`, background: row.color }"
        />
        <div
          v-if="threshold !== undefined"
          class="absolute inset-y-0 w-px bg-slate-400/50"
          :style="{ left: `${threshold * 100}%` }"
        />
      </div>
    </div>
  </div>
</template>
