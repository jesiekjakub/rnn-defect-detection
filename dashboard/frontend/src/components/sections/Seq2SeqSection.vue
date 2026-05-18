<script setup lang="ts">
import StackedDerivatives from '@/components/charts/StackedDerivatives.vue';
import RegionTimeline from '@/components/charts/RegionTimeline.vue';
import { useSampleStore } from '@/stores/sample';

const sample = useSampleStore();
</script>

<template>
  <section v-if="sample.compare && sample.sequence" class="space-y-6 p-6">
    <div class="card p-5">
      <p class="section-title">Original · residual · velocity</p>
      <StackedDerivatives
        :original="sample.sequence.x"
        :residual="sample.compare.seq2seq.residual"
        :velocity="sample.compare.seq2seq.velocity"
        :accepted-regions="sample.compare.seq2seq.accepted_regions"
      />
    </div>

    <div class="card p-5">
      <p class="section-title">Region proposal → verification → accepted</p>
      <RegionTimeline
        :candidates="sample.compare.seq2seq.candidates"
        :verified="sample.compare.seq2seq.verified"
        :accepted="sample.compare.seq2seq.accepted_regions"
        :seq-len="sample.sequence.x.length"
      />
    </div>
  </section>
  <section v-else class="grid place-items-center p-12 text-sm text-slate-500">
    Run a sample on the Live Demo tab first.
  </section>
</template>
