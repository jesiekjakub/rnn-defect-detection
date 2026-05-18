<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { api } from '@/api/client';
import type { ArchitectureGraph, ArchitectureNode } from '@/api/types';

const graphs = ref<ArchitectureGraph[] | null>(null);
const activeNode = ref<ArchitectureNode | null>(null);

onMounted(async () => {
  const response = await api.architecture();
  graphs.value = response.approaches;
});

function pick(node: ArchitectureNode) {
  activeNode.value = node;
}
</script>

<template>
  <section class="space-y-8 p-6">
    <div v-for="graph in graphs ?? []" :key="graph.name" class="card p-5">
      <p class="text-base font-semibold text-slate-100">{{ graph.name }}</p>
      <div class="mt-5 flex flex-wrap gap-3">
        <button
          v-for="node in graph.nodes"
          :key="node.id"
          class="group rounded-xl border border-slate-800 bg-slate-950/40 p-4 text-left transition hover:border-cyan-400/60 hover:bg-cyan-500/5"
          @click="pick(node)"
        >
          <p class="text-[10px] uppercase tracking-wider text-slate-500">{{ node.id }}</p>
          <p class="mt-1 font-medium text-slate-100">{{ node.label }}</p>
          <p v-if="node.notes" class="mt-2 text-xs text-slate-400">{{ node.notes }}</p>
          <p class="mt-3 text-[11px] text-cyan-200/80 transition group-hover:text-cyan-200">
            {{ node.layers.length }} layer{{ node.layers.length === 1 ? '' : 's' }} →
          </p>
        </button>
      </div>

      <div class="mt-5 flex flex-wrap gap-2 text-xs text-slate-400">
        <span
          v-for="(edge, i) in graph.edges"
          :key="i"
          class="rounded-full border border-slate-800 bg-slate-950/40 px-3 py-1"
        >
          {{ edge.src }} → {{ edge.dst }} <span v-if="edge.label" class="text-slate-500">· {{ edge.label }}</span>
        </span>
      </div>
    </div>

    <div v-if="activeNode" class="card sticky bottom-6 mx-auto max-w-3xl p-5">
      <div class="flex items-center justify-between">
        <p class="text-base font-semibold text-slate-100">{{ activeNode.label }}</p>
        <button class="text-xs text-slate-400 hover:text-slate-100" @click="activeNode = null">close</button>
      </div>
      <p v-if="activeNode.notes" class="mt-2 text-xs text-slate-400">{{ activeNode.notes }}</p>
      <ul class="mt-4 space-y-2 text-sm">
        <li
          v-for="layer in activeNode.layers"
          :key="layer.name"
          class="rounded-lg border border-slate-800 bg-slate-950/40 p-3"
        >
          <p class="flex items-center justify-between">
            <span class="text-slate-100">{{ layer.name }}</span>
            <span class="text-[10px] uppercase tracking-wider text-cyan-200">{{ layer.kind }}</span>
          </p>
          <pre class="mt-2 overflow-auto text-xs text-slate-400">{{ JSON.stringify(layer.params, null, 2) }}</pre>
        </li>
      </ul>
    </div>
  </section>
</template>
