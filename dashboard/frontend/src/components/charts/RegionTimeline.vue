<script setup lang="ts">
import { computed } from 'vue';
import { DEFECT_COLORS } from '@/api/types';
import type { CandidateRegion, VerifiedRegion } from '@/api/types';

const props = defineProps<{
  candidates: CandidateRegion[];
  verified: VerifiedRegion[];
  accepted: VerifiedRegion[];
  seqLen: number;
}>();

type Row = {
  label: string;
  start: number;
  end: number;
  color: string;
  stage: 'candidate' | 'verified' | 'accepted';
  detail: string;
};

const rows = computed<Row[]>(() => {
  const candidateRows: Row[] = props.candidates.map((c) => ({
    label: c.source === 'residual' ? 'Residual peak' : 'Velocity pair',
    start: c.start,
    end: c.end,
    color: 'rgba(148,163,184,0.45)',
    stage: 'candidate',
    detail: `${c.source} · t=${c.start}–${c.end}`,
  }));
  const verifiedRows: Row[] = props.verified.map((v) => ({
    label: v.defect_name,
    start: v.start,
    end: v.end,
    color: hexWithAlpha(DEFECT_COLORS[v.defect_index], v.consensus_pass ? 0.7 : 0.25),
    stage: 'verified',
    detail: `local p=${(v.local_probability * 100).toFixed(1)}% · consensus ${v.consensus_pass ? '✓' : '✗'}`,
  }));
  const acceptedRows: Row[] = props.accepted.map((a) => ({
    label: a.defect_name,
    start: a.start,
    end: a.end,
    color: DEFECT_COLORS[a.defect_index],
    stage: 'accepted',
    detail: `accepted · t=${a.start}–${a.end} · p=${(a.local_probability * 100).toFixed(1)}%`,
  }));
  return [...candidateRows, ...verifiedRows, ...acceptedRows];
});

function widthPct(start: number, end: number) {
  return `${Math.max(2, ((end - start + 1) / props.seqLen) * 100)}%`;
}
function leftPct(start: number) {
  return `${(start / props.seqLen) * 100}%`;
}
function hexWithAlpha(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const STAGE_LABEL: Record<Row['stage'], string> = {
  candidate: 'Proposed candidates',
  verified: 'Local verification',
  accepted: 'Accepted (consensus)',
};

const grouped = computed(() => {
  const groups: Record<Row['stage'], Row[]> = { candidate: [], verified: [], accepted: [] };
  for (const row of rows.value) groups[row.stage].push(row);
  return (['candidate', 'verified', 'accepted'] as Row['stage'][])
    .filter((stage) => groups[stage].length > 0)
    .map((stage) => ({ stage, label: STAGE_LABEL[stage], rows: groups[stage] }));
});
</script>

<template>
  <div class="space-y-4">
    <div v-for="group in grouped" :key="group.stage">
      <p class="section-title mb-2">{{ group.label }}</p>
      <div class="space-y-2">
        <div
          v-for="(row, i) in group.rows"
          :key="`${group.stage}-${i}`"
          class="relative h-7 rounded-md border border-slate-800/60 bg-slate-900/50"
        >
          <div
            class="absolute inset-y-1 rounded transition-all duration-300"
            :style="{ left: leftPct(row.start), width: widthPct(row.start, row.end), background: row.color }"
          />
          <div class="absolute inset-y-0 left-2 flex items-center gap-3 text-xs text-slate-200">
            <span class="font-medium">{{ row.label }}</span>
            <span class="font-mono text-slate-400">{{ row.detail }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
