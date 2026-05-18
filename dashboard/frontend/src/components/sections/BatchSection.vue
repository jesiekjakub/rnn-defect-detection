<script setup lang="ts">
import { onMounted } from 'vue';
import { api } from '@/api/client';
import { useBatchStore } from '@/stores/batch';
import { useSampleStore } from '@/stores/sample';
import { DEFECT_COLORS, DEFECT_NAMES, NUM_DEFECT_TYPES } from '@/api/types';
import CoOccurrenceHeatmap from '@/components/charts/CoOccurrenceHeatmap.vue';

const batch = useBatchStore();
const sample = useSampleStore();

onMounted(() => {
  if (!batch.rows) void batch.refresh();
});

function setFilter(predDefect?: number) {
  batch.setFilters({ predDefect });
  void batch.refresh();
}

function setConfidence(min: number, max: number) {
  batch.setFilters({ confidenceMin: min, confidenceMax: max });
  void batch.refresh();
}

function toggleAgreement() {
  batch.setFilters({ agreementOnly: !batch.agreementOnly });
  void batch.refresh();
}

async function openSample(sampleId: number) {
  const compare = await api.batchSample(sampleId);
  await sample.loadPrecomputed(compare);
}

function dotsFor(values: number[]): { color: string; on: boolean }[] {
  return values.map((v, i) => ({ color: DEFECT_COLORS[i], on: v === 1 }));
}
</script>

<template>
  <section class="space-y-6 p-6">
    <div class="card p-5">
      <div class="flex flex-wrap items-center gap-3">
        <p class="section-title">Filters</p>
        <div class="flex flex-wrap gap-2">
          <button
            :class="[
              'rounded-full border px-3 py-1 text-xs transition',
              batch.predDefect === undefined
                ? 'border-cyan-400 bg-cyan-500/15 text-cyan-100'
                : 'border-slate-700 text-slate-300 hover:border-cyan-400',
            ]"
            @click="setFilter(undefined)"
          >
            all
          </button>
          <button
            v-for="i in NUM_DEFECT_TYPES"
            :key="i"
            :class="[
              'rounded-full border px-3 py-1 text-xs transition',
              batch.predDefect === i - 1
                ? 'border-cyan-400 bg-cyan-500/15 text-cyan-100'
                : 'border-slate-700 text-slate-300 hover:border-cyan-400',
            ]"
            @click="setFilter(i - 1)"
          >
            <span class="mr-1 inline-block h-2 w-2 rounded-full" :style="{ background: DEFECT_COLORS[i - 1] }" />
            {{ DEFECT_NAMES[i - 1] }}
          </button>
        </div>
        <div class="ml-auto flex items-center gap-2 text-xs text-slate-400">
          <label class="flex items-center gap-2">
            <input
              type="checkbox"
              class="h-3.5 w-3.5 accent-cyan-400"
              :checked="batch.agreementOnly"
              @change="toggleAgreement"
            />
            agreement only
          </label>
        </div>
      </div>
      <div class="mt-4 flex items-center gap-4 text-xs text-slate-400">
        <span>max-prob ∈ [{{ batch.confidenceMin.toFixed(2) }}, {{ batch.confidenceMax.toFixed(2) }}]</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          :value="batch.confidenceMin"
          class="flex-1 accent-cyan-400"
          @change="setConfidence(Number(($event.target as HTMLInputElement).value), batch.confidenceMax)"
        />
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          :value="batch.confidenceMax"
          class="flex-1 accent-cyan-400"
          @change="setConfidence(batch.confidenceMin, Number(($event.target as HTMLInputElement).value))"
        />
      </div>
    </div>

    <div class="grid grid-cols-1 gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <div class="card p-3">
        <div class="overflow-auto scrollbar-thin">
          <table class="min-w-full text-xs">
            <thead class="border-b border-slate-800 text-slate-400">
              <tr>
                <th class="px-3 py-2 text-left">#</th>
                <th class="px-3 py-2 text-left">Truth</th>
                <th class="px-3 py-2 text-left">A1 pred</th>
                <th class="px-3 py-2 text-left">A2 pred</th>
                <th class="px-3 py-2 text-left">Agreement</th>
                <th class="px-3 py-2 text-left">A1 max prob</th>
                <th class="px-3 py-2 text-left">A2 max prob</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in batch.rows?.rows ?? []"
                :key="row.sample_id"
                class="cursor-pointer border-b border-slate-900 transition hover:bg-cyan-500/5"
                @click="openSample(row.sample_id)"
              >
                <td class="px-3 py-2 tabular text-slate-200">{{ row.sample_id }}</td>
                <td class="px-3 py-2">
                  <span class="flex gap-1">
                    <span
                      v-for="(d, i) in dotsFor(row.y_true)"
                      :key="`t-${i}`"
                      class="h-2 w-2 rounded-full"
                      :class="d.on ? '' : 'opacity-20'"
                      :style="{ background: d.color }"
                    />
                  </span>
                </td>
                <td class="px-3 py-2">
                  <span class="flex gap-1">
                    <span
                      v-for="(d, i) in dotsFor(row.pred_a1)"
                      :key="`a1-${i}`"
                      class="h-2 w-2 rounded-full"
                      :class="d.on ? '' : 'opacity-20'"
                      :style="{ background: d.color }"
                    />
                  </span>
                </td>
                <td class="px-3 py-2">
                  <span class="flex gap-1">
                    <span
                      v-for="(d, i) in dotsFor(row.pred_a2)"
                      :key="`a2-${i}`"
                      class="h-2 w-2 rounded-full"
                      :class="d.on ? '' : 'opacity-20'"
                      :style="{ background: d.color }"
                    />
                  </span>
                </td>
                <td class="px-3 py-2">
                  <span
                    :class="[
                      'rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide',
                      row.agreement ? 'bg-emerald-500/15 text-emerald-200' : 'bg-rose-500/15 text-rose-200',
                    ]"
                  >
                    {{ row.agreement ? 'agree' : 'differ' }}
                  </span>
                </td>
                <td class="px-3 py-2 tabular text-slate-300">{{ Math.max(...row.probs_a1).toFixed(3) }}</td>
                <td class="px-3 py-2 tabular text-slate-300">{{ Math.max(...row.probs_a2).toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="px-3 py-3 text-xs text-slate-500">
          Showing {{ batch.rows?.rows.length ?? 0 }} of {{ batch.rows?.total ?? 0 }} matches. Click any row to inspect.
        </p>
      </div>

      <div class="card p-5">
        <p class="section-title mb-3">Predicted defect co-occurrence (A1)</p>
        <CoOccurrenceHeatmap v-if="batch.rows" :rows="batch.rows.rows" approach="attention" />
        <p v-else class="text-xs text-slate-500">Loading rows…</p>
      </div>
    </div>
  </section>
</template>
