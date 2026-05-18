import { defineStore } from 'pinia';

export const useUiStore = defineStore('ui', {
  state: () => ({
    sidebarCollapsed: false,
    pinnedTimestep: null as number | null,
    activeDefectFilter: null as number | null,
    health: {
      models_loaded: false,
      cache_ready: false,
      device: 'unknown',
    },
  }),
  actions: {
    toggleSidebar() {
      this.sidebarCollapsed = !this.sidebarCollapsed;
    },
    pinTimestep(t: number | null) {
      this.pinnedTimestep = t;
    },
    setHealth(payload: { models_loaded: boolean; cache_ready: boolean; device: string }) {
      this.health = payload;
    },
  },
});
