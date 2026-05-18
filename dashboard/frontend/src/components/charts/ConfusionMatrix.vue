<script setup lang="ts">
import { computed } from 'vue';
import type { PerClassMetric } from '@/api/types';
import { DEFECT_NAMES } from '@/api/types';

const props = defineProps<{
  perClass: PerClassMetric[];
}>();

type Cell = {
  name: string;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  total: number;
};

const cells = computed<Cell[]>(() =>
  props.perClass.map((m, i) => ({
    name: DEFECT_NAMES[i],
    tp: m.true_positive,
    fp: m.false_positive,
    fn: m.false_negative,
    tn: m.true_negative,
    total: m.true_positive + m.false_positive + m.false_negative + m.true_negative,
  })),
);

function pct(value: number, total: number) {
  return total === 0 ? '0%' : `${((value / total) * 100).toFixed(1)}%`;
}
</script>

<template>
  <div class="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
    <div
      v-for="cell in cells"
      :key="cell.name"
      class="card overflow-hidden p-4"
    >
      <p class="text-xs font-medium tracking-wide text-slate-400">{{ cell.name }}</p>
      <div class="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg bg-slate-800">
        <div class="bg-emerald-500/15 px-3 py-3">
          <p class="text-[10px] uppercase tracking-wide text-emerald-200">true positive</p>
          <p class="mt-1 tabular text-lg text-emerald-100">{{ cell.tp }}</p>
          <p class="text-[10px] text-emerald-300/80">{{ pct(cell.tp, cell.total) }}</p>
        </div>
        <div class="bg-orange-500/15 px-3 py-3">
          <p class="text-[10px] uppercase tracking-wide text-orange-200">false positive</p>
          <p class="mt-1 tabular text-lg text-orange-100">{{ cell.fp }}</p>
          <p class="text-[10px] text-orange-300/80">{{ pct(cell.fp, cell.total) }}</p>
        </div>
        <div class="bg-rose-500/15 px-3 py-3">
          <p class="text-[10px] uppercase tracking-wide text-rose-200">false negative</p>
          <p class="mt-1 tabular text-lg text-rose-100">{{ cell.fn }}</p>
          <p class="text-[10px] text-rose-300/80">{{ pct(cell.fn, cell.total) }}</p>
        </div>
        <div class="bg-slate-700/40 px-3 py-3">
          <p class="text-[10px] uppercase tracking-wide text-slate-300">true negative</p>
          <p class="mt-1 tabular text-lg text-slate-100">{{ cell.tn }}</p>
          <p class="text-[10px] text-slate-400">{{ pct(cell.tn, cell.total) }}</p>
        </div>
      </div>
    </div>
  </div>
</template>
