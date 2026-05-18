<script setup lang="ts">
import { ref } from 'vue';
import { UploadCloud, X } from 'lucide-vue-next';
import { useUploadStore } from '@/stores/upload';

const upload = useUploadStore();
const dragOver = ref(false);
const fileInput = ref<HTMLInputElement | null>(null);

function handleFiles(files: FileList | null) {
  if (!files || files.length === 0) return;
  void upload.ingest(files[0]);
}

function onDrop(event: DragEvent) {
  event.preventDefault();
  dragOver.value = false;
  handleFiles(event.dataTransfer?.files ?? null);
}
</script>

<template>
  <div
    :class="[
      'card flex flex-col items-center gap-3 border-dashed p-8 text-center transition',
      dragOver ? 'border-cyan-400 bg-cyan-500/5' : 'border-slate-700/60',
    ]"
    @dragover.prevent="dragOver = true"
    @dragleave="dragOver = false"
    @drop="onDrop"
  >
    <UploadCloud class="h-9 w-9 text-cyan-300" />
    <div>
      <p class="text-base font-semibold text-slate-100">Drop a CSV or JSON file</p>
      <p class="mt-1 text-xs text-slate-400">
        CSV columns: <code class="font-mono text-slate-200">sequence_id, t, sensor_0..2, defect_0..4?</code>
        · JSON: array of <code class="font-mono text-slate-200">{ x: number[][], y_true?: number[] }</code>
      </p>
    </div>
    <button
      class="rounded-lg border border-slate-700 px-4 py-2 text-xs font-medium text-slate-200 transition hover:border-cyan-400 hover:text-cyan-200"
      @click="fileInput?.click()"
    >
      Pick a file
    </button>
    <input
      ref="fileInput"
      type="file"
      class="hidden"
      accept=".csv,.json,application/json,text/csv"
      @change="handleFiles(($event.target as HTMLInputElement).files)"
    />
    <p v-if="upload.loading" class="text-xs text-cyan-200">parsing…</p>
    <p v-if="upload.error" class="text-xs text-rose-300">{{ upload.error }}</p>
    <ul v-if="upload.warnings.length" class="mt-1 space-y-1 text-left text-[11px] text-amber-300">
      <li v-for="warn in upload.warnings" :key="warn">⚠ {{ warn }}</li>
    </ul>
    <button
      v-if="upload.sequences.length"
      class="mt-2 flex items-center gap-1 text-xs text-slate-400 transition hover:text-slate-200"
      @click="upload.clear()"
    >
      <X class="h-3 w-3" /> Clear upload
    </button>
  </div>
</template>
