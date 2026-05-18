<script setup lang="ts">
import { computed } from 'vue';
import { useUiStore } from '@/stores/ui';
import { useSampleStore } from '@/stores/sample';
import { Activity, Github, Loader2, ShieldCheck, AlertTriangle } from 'lucide-vue-next';

const ui = useUiStore();
const sample = useSampleStore();

const statusLabel = computed(() => {
  if (sample.loading) return 'Running inference';
  if (!ui.health.models_loaded) return 'Models not loaded';
  if (!ui.health.cache_ready) return 'Warming analytics cache';
  return `Ready · ${ui.health.device}`;
});

const statusTone = computed(() => {
  if (!ui.health.models_loaded) return 'text-amber-400';
  if (!ui.health.cache_ready) return 'text-amber-300';
  return 'text-emerald-300';
});
</script>

<template>
  <header
    class="flex items-center justify-between border-b border-slate-800/60 bg-slate-950/60 px-6 py-3 backdrop-blur"
  >
    <div class="flex items-center gap-3">
      <div class="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-cyan-400/20 to-fuchsia-500/10">
        <Activity class="h-5 w-5 text-cyan-300" />
      </div>
      <div>
        <p class="text-sm font-semibold tracking-wide text-slate-100">RNN Defect Detection</p>
        <p class="text-xs text-slate-400">Interactive companion dashboard</p>
      </div>
    </div>

    <div class="flex items-center gap-4">
      <div :class="['flex items-center gap-2 text-xs font-medium', statusTone]">
        <Loader2 v-if="sample.loading" class="h-4 w-4 animate-spin" />
        <AlertTriangle v-else-if="!ui.health.models_loaded" class="h-4 w-4" />
        <ShieldCheck v-else class="h-4 w-4" />
        <span>{{ statusLabel }}</span>
      </div>
      <a
        href="https://github.com/jesiekjakub/rnn-defect-detection"
        target="_blank"
        rel="noopener"
        class="flex h-9 w-9 items-center justify-center rounded-xl border border-slate-800/80 text-slate-400 transition hover:border-slate-700 hover:text-slate-100"
        aria-label="View source on GitHub"
      >
        <Github class="h-4 w-4" />
      </a>
    </div>
  </header>
</template>
