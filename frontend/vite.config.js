import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'    // 新加

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  resolve: {                // 新加
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    proxy: {
      '/predict': { target: 'http://localhost:3000', changeOrigin: true },
      '/stations': { target: 'http://localhost:3000', changeOrigin: true },
      '/dispatch': { target: 'http://localhost:3000', changeOrigin: true },
      '/schedule': { target: 'http://localhost:3000', changeOrigin: true },
      '/search': { target: 'http://localhost:3000', changeOrigin: true },
      '/suggestions': { target: 'http://localhost:3000', changeOrigin: true },
      '/guide': { target: 'http://localhost:3000', changeOrigin: true },
      '/statistics': { target: 'http://localhost:3000', changeOrigin: true },
      '/admin': { target: 'http://localhost:3000', changeOrigin: true },
      '/reset': { target: 'http://localhost:3000', changeOrigin: true }
    }
  }
  
})
