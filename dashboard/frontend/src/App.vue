<script setup lang="ts">
import { onMounted } from 'vue';
import AppHeader from '@/components/layout/AppHeader.vue';
import AppSidebar from '@/components/layout/AppSidebar.vue';
import { api } from '@/api/client';
import { useUiStore } from '@/stores/ui';
import { useSampleStore } from '@/stores/sample';

const ui = useUiStore();
const sample = useSampleStore();

onMounted(async () => {
  try {
    const health = await api.health();
    ui.setHealth({
      models_loaded: health.models_loaded,
      cache_ready: health.cache_ready,
      device: health.device,
    });
    if (health.models_loaded) {
      await sample.generateAndPredict();
    }
  } catch (err) {
    console.error('health check failed', err);
  }
});
</script>

<template>
  <div class="flex min-h-screen bg-slate-950 text-slate-100">
    <AppSidebar />
    <div class="flex flex-1 flex-col">
      <AppHeader />
      <main class="flex-1 overflow-y-auto scrollbar-thin">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </main>
    </div>
  </div>
</template>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.18s ease, transform 0.18s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(4px);
}
</style>
