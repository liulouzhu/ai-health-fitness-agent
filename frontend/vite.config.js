import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/chat': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/profile': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/daily_stats': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/history': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/uploads': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      '/upload-image': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})
