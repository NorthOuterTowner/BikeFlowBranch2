import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/stations': {
        target: 'http://localhost:3000',  // 这里替换成你后端真实地址
        changeOrigin: true
        // 如果后端接口就是 /stations/xxx，不需要 rewrite
        // 如果后端只提供 /xxx，就需要加：
        // rewrite: (path) => path.replace(/^\/stations/, '')
      }
    }
  }
})

