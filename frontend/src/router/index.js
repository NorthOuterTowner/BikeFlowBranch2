import { createRouter, createWebHistory } from 'vue-router'

// 引入你要显示的页面组件
import Dashboard from '../views/dashboard/dashboard.vue'
import Login from '../views/login.vue'
import Register from '../views/register.vue'
import MapView from '../views/dashboard/mapView.vue'
import Predict from '../views/dashboard/predict.vue'
import Settings from '../views/dashboard/Settings.vue'
import Profile from '../views/dashboard/profile.vue'
import schedule from '../views/dashboard/schedule.vue'
import dispatch from '../views/dashboard/dispatch.vue'
import guide from '../views/dashboard/guide.vue'
import deepseek from '../views/dashboard/deepseek.vue'
import statistics from '../views/dashboard/statistics.vue'
import faq from '../views/dashboard/faq.vue'

const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', component: Login },
  { path: '/register', component: Register },
   
  {
    path: '/dashboard',
    component: Dashboard,
    children: [
      { path: '', redirect: '/dashboard/mapView' },
      { path: 'mapView', component: MapView },
      { path: 'predict', component: Predict },
      { path: 'settings', component: Settings },
      { path: 'profile', component: Profile },
      { path: 'schedule', component: schedule },
      { path: 'dispatch', component: dispatch },
      { path: 'guide', component: guide },
      {path: 'faq', component: faq },
      { path: 'deepseek', component: deepseek },
      { path: 'statistics', component: statistics },
    ]
  },
]

//  先创建 router
const router = createRouter({
  history: createWebHistory(),
  routes
})

// 再添加导航守卫
router.beforeEach((to, from, next) => {
  const publicPages = ['/login', '/register']
  const authRequired = !publicPages.includes(to.path)
  const loggedIn = sessionStorage.getItem('token') 

  if (authRequired && !loggedIn) {
    return next('/login')
  }
  next()
})

export default router
