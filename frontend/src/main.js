import { createApp } from 'vue'
import { createPinia } from 'pinia'
import naive from 'naive-ui'
//import './style.css'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'

//createApp(App).mount('#app')

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)   // 注册路由插件

app.mount('#app')
app.use(naive) // 注册 Naive UI 插件

app.use(ElementPlus) // 注册 Element Plus 插件
