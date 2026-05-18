import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_BASE || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.VITE_API_BASE || 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
