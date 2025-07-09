import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/predict': {
        target: 'http://localhost:3000',
        changeOrigin: true
        // 不需要 rewrite，前端写 /predict/xxx，后端也就是 /predict/xxx
      },
      '/stations': {
        target: 'http://localhost:3000',
        changeOrigin: true
      }
    } 
    
  }
})

