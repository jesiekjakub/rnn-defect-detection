<script setup lang="ts">
import { computed } from 'vue';
import { RouterLink, useRoute } from 'vue-router';
import {
  Activity, Columns, Database, Eye, Layers, LineChart, Network, Orbit, Play,
  SlidersHorizontal, Upload, ChevronLeft, ChevronRight,
} from 'lucide-vue-next';
import { navEntries } from '@/router';
import { useUiStore } from '@/stores/ui';

const ui = useUiStore();
const route = useRoute();

const ICONS: Record<string, typeof Activity> = {
  activity: Activity,
  upload: Upload,
  database: Database,
  eye: Eye,
  layers: Layers,
  columns: Columns,
  'sliders-horizontal': SlidersHorizontal,
  orbit: Orbit,
  play: Play,
  'line-chart': LineChart,
  network: Network,
};

const items = computed(() =>
  navEntries.map((entry) => ({
    ...entry,
    icon: ICONS[entry.iconKey] ?? Activity,
    active: route.path === entry.path,
  })),
);
</script>

<template>
  <aside
    :class="[
      'flex shrink-0 flex-col border-r border-slate-800/60 bg-slate-950/80 backdrop-blur transition-[width] duration-200',
      ui.sidebarCollapsed ? 'w-16' : 'w-60',
    ]"
  >
    <button
      class="m-3 flex h-9 items-center justify-center rounded-lg border border-slate-800 text-slate-400 transition hover:border-slate-700 hover:text-slate-100"
      :aria-label="ui.sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'"
      @click="ui.toggleSidebar()"
    >
      <ChevronLeft v-if="!ui.sidebarCollapsed" class="h-4 w-4" />
      <ChevronRight v-else class="h-4 w-4" />
    </button>

    <nav class="flex-1 space-y-1 px-2">
      <RouterLink
        v-for="item in items"
        :key="item.path"
        :to="item.path"
        :class="[
          'group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition',
          item.active
            ? 'bg-cyan-500/10 text-cyan-200 ring-1 ring-cyan-500/30'
            : 'text-slate-400 hover:bg-slate-800/60 hover:text-slate-100',
        ]"
        :title="ui.sidebarCollapsed ? item.label : undefined"
      >
        <component :is="item.icon" class="h-4 w-4 shrink-0" />
        <span v-if="!ui.sidebarCollapsed" class="truncate">{{ item.label }}</span>
      </RouterLink>
    </nav>

    <div v-if="!ui.sidebarCollapsed" class="border-t border-slate-800/60 p-4 text-xs text-slate-500">
      <p class="font-mono">device · {{ ui.health.device }}</p>
      <p class="mt-1 font-mono">cache · {{ ui.health.cache_ready ? 'ready' : 'warming' }}</p>
    </div>
  </aside>
</template>
