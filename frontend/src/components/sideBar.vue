<template>
  <aside class="sidebar">
    <nav>
      <div class="menu-section">
        <h1 @click="toggleSection('predict')" class="menu-title">
          预测显示
          <span class="arrow">{{ openSections.predict ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.predict">
          <li><router-link to="/dashboard/mapView">地图</router-link></li>
          <li><router-link to="/dashboard/predict">预测结果</router-link></li>
        </ul>
      </div>

      <div class="menu-section">
        <h1 @click="toggleSection('schedule')" class="menu-title">
          调度显示
          <span class="arrow">{{ openSections.schedule ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.schedule">
          <li><router-link to="/dashboard/schedule">调度页面</router-link></li>
          <li><router-link to="/dashboard/dispatch">调度详情</router-link></li>
        </ul>
      </div>

      <div class="menu-section">
        <h1 @click="toggleSection('settings')" class="menu-title">
          设置
          <span class="arrow">{{ openSections.settings ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.settings">
          <li><router-link to="/dashboard/settings">系统设置</router-link></li>
          <li><router-link to="/dashboard/profile">个人信息</router-link></li>
        </ul>
      </div>
    </nav>
  </aside>
</template>

<script setup>
import { reactive } from 'vue'

const openSections = reactive({
  predict: false,  // 默认展开或折叠
  schedule: false,
  settings: false,
})

function toggleSection(section) {
  openSections[section] = !openSections[section]
}
</script>

<style scoped>
.sidebar {
  width: 200px;
  background-color: #001f4d; 
  padding: 1rem;
  color: #cce0ff; 
}

.menu-title {
  font-size: 1rem;
  margin: 1rem 0 0.5rem;
  cursor: pointer;
  user-select: none;
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #99c2ff; /* 较亮蓝色标题 */
}

.arrow {
  font-size: 0.8rem;
  color: #99c2ff;
}

ul {
  list-style: none;
  padding-left: 1rem;
  margin: 0;
}

li {
  margin-bottom: 0.5rem;
}

li a {
  color: white;
  text-decoration: none;
}

li a.router-link-exact-active {
  font-weight: bold;
  color: white;
}

li a:hover {
  text-decoration: underline;
  color: #ccc;
}

</style>
