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
      '/predict': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/stations': {
        target: 'http://localhost:3000',
        changeOrigin: true
      },
      '/dispatch': {
        target: 'http://localhost:3000',  // 换成后端实际地址和端口
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
      },
      '/guide': { 
        target: 'http://localhost:3000', 
        changeOrigin: true 
      }, 
      '/statistics': { 
        target: 'http://localhost:3000', 
        changeOrigin: true 
      },
    } 
  }
})
