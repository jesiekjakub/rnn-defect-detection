import { createRouter, createWebHashHistory, type RouteRecordRaw } from 'vue-router';

import LiveDemoSection from '@/components/sections/LiveDemoSection.vue';
import UploadSection from '@/components/sections/UploadSection.vue';
import BatchSection from '@/components/sections/BatchSection.vue';
import AttentionSection from '@/components/sections/AttentionSection.vue';
import Seq2SeqSection from '@/components/sections/Seq2SeqSection.vue';
import ComparisonSection from '@/components/sections/ComparisonSection.vue';
import ThresholdSection from '@/components/sections/ThresholdSection.vue';
import LatentSection from '@/components/sections/LatentSection.vue';
import StreamingSection from '@/components/sections/StreamingSection.vue';
import PerformanceSection from '@/components/sections/PerformanceSection.vue';
import ArchitectureSection from '@/components/sections/ArchitectureSection.vue';

export type NavEntry = {
  path: string;
  label: string;
  short: string;
  iconKey: string;
};

export const navEntries: NavEntry[] = [
  { path: '/demo', label: 'Live Demo', short: 'Demo', iconKey: 'activity' },
  { path: '/upload', label: 'Upload & Analyze', short: 'Upload', iconKey: 'upload' },
  { path: '/batch', label: 'Batch Explorer', short: 'Batch', iconKey: 'database' },
  { path: '/approach-1', label: 'Approach 1 · Attention', short: 'A1', iconKey: 'eye' },
  { path: '/approach-2', label: 'Approach 2 · Seq2Seq', short: 'A2', iconKey: 'layers' },
  { path: '/compare', label: 'Comparison', short: 'Compare', iconKey: 'columns' },
  { path: '/threshold', label: 'Threshold Lab', short: 'Threshold', iconKey: 'sliders-horizontal' },
  { path: '/latent', label: 'Latent Space', short: 'Latent', iconKey: 'orbit' },
  { path: '/stream', label: 'Streaming Replay', short: 'Stream', iconKey: 'play' },
  { path: '/performance', label: 'Performance', short: 'Perf', iconKey: 'line-chart' },
  { path: '/architecture', label: 'Architecture', short: 'Arch', iconKey: 'network' },
];

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/demo' },
  { path: '/demo', component: LiveDemoSection },
  { path: '/upload', component: UploadSection },
  { path: '/batch', component: BatchSection },
  { path: '/approach-1', component: AttentionSection },
  { path: '/approach-2', component: Seq2SeqSection },
  { path: '/compare', component: ComparisonSection },
  { path: '/threshold', component: ThresholdSection },
  { path: '/latent', component: LatentSection },
  { path: '/stream', component: StreamingSection },
  { path: '/performance', component: PerformanceSection },
  { path: '/architecture', component: ArchitectureSection },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
