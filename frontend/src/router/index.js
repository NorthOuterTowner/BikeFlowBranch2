import { createRouter, createWebHistory } from 'vue-router'

// 引入你要显示的页面组件
import Dashboard from '../views/dashboard/dashboard.vue'
import Login from '../views/login.vue'
import Register from '../views/register.vue'
import MapView from '../views/dashboard/mapView.vue'
import Predict from '../views/dashboard/Predict.vue'
import Settings from '../views/dashboard/Settings.vue'
import Profile from '../views/dashboard/profile.vue'
import schedule from '../views/dashboard/schedule.vue'
import dispatch from '../views/dashboard/dispatch.vue'
import guide from '../views/dashboard/guide.vue'

const routes = [
  { path: '/', redirect: '/login' },    // 默认跳到 dashboard
  { path: '/login', component: Login },
  { path: '/register', component: Register },
  { path: '/dashboard', component: Dashboard ,
    children: [
        { path: '', redirect: '/dashboard/mapView' },  // 默认进来显示地图页
        { path: 'mapView', component: MapView },
        { path: 'predict', component: Predict },
        { path: 'settings', component: Settings },
        { path: 'profile', component: Profile },
        {path: 'schedule', component: schedule },
        {path:'dispatch', component: dispatch },
        {path: 'guide', component: guide }
    ]
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
