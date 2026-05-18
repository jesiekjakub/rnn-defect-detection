<script setup lang="ts">
import { computed } from 'vue';
import { DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';

const props = defineProps<{
  predA1: boolean[];
  predA2: boolean[];
  truth?: number[] | null;
}>();

type Cell = { label: string; tone: string; helper: string };

const cells = computed<Cell[]>(() => {
  return Array.from({ length: NUM_DEFECT_TYPES }, (_, i) => {
    const a1 = props.predA1[i];
    const a2 = props.predA2[i];
    const truth = props.truth?.[i];
    let tone = 'bg-slate-900/50 text-slate-400';
    let label = 'both clear';
    if (a1 && a2) {
      tone = 'bg-cyan-500/15 text-cyan-200 ring-1 ring-cyan-500/30';
      label = 'both detect';
    } else if (a1 && !a2) {
      tone = 'bg-orange-500/15 text-orange-200 ring-1 ring-orange-500/30';
      label = 'A1 only';
    } else if (!a1 && a2) {
      tone = 'bg-fuchsia-500/15 text-fuchsia-200 ring-1 ring-fuchsia-500/30';
      label = 'A2 only';
    }
    const helper = truth === undefined ? '' : truth === 1 ? 'truth: positive' : 'truth: negative';
    return { label, tone, helper };
  });
});
</script>

<template>
  <div class="grid grid-cols-5 gap-3">
    <div
      v-for="(cell, i) in cells"
      :key="i"
      :class="['rounded-lg px-3 py-3 text-center transition', cell.tone]"
    >
      <p class="text-[10px] font-medium uppercase tracking-wider text-slate-400">{{ DEFECT_NAMES[i] }}</p>
      <p class="mt-1 text-sm font-semibold">{{ cell.label }}</p>
      <p v-if="cell.helper" class="mt-1 text-[11px] text-slate-500">{{ cell.helper }}</p>
    </div>
  </div>
</template>
