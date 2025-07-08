import { createRouter, createWebHistory } from 'vue-router'

// 引入你要显示的页面组件
import Dashboard from '../views/dashboard/dashboard.vue'
import Login from '../views/login.vue'
import MapView from '../views/dashboard/MapView.vue'
import Predict from '../views/dashboard/Predict.vue'
//import Settings from '../views/dashboard/Settings.vue'
//import Profile from '../views/dashboard/Profile.vue'

const routes = [
  { path: '/', redirect: '/dashboard' },    // 默认跳到 dashboard
  { path: '/login', component: Login },
  { path: '/dashboard', component: Dashboard ,
    children: [
        { path: '', redirect: '/dashboard/mapView' },  // 默认进来显示地图页
        { path: 'mapView', component: MapView },
        { path: 'predict', component: Predict },
       // { path: 'settings', component: Settings },
       // { path: 'profile', component: Profile },
    ]
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
