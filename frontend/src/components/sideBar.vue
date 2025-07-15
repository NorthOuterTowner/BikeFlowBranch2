<template>
  <aside class="sidebar" :class="{ 'collapsed': collapsed }">
    <!-- 收起/展开按钮 -->
    <div class="toggle-btn" @click="handleToggle">
      <div class="hamburger-icon">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
    
    <nav style="margin-top: 50px;">
      <div class="menu-section">
        <h1 @click="toggleSection('predict')" class="menu-title">
          <span class="title-text" v-show="!collapsed">预测显示</span>
          <span class="arrow" v-show="!collapsed">{{ openSections.predict ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.predict && !collapsed">
          <li><router-link to="/dashboard/mapView">预测地图</router-link></li>
          <li><router-link to="/dashboard/predict">预测结果</router-link></li>
        </ul>
      </div>
      
      <div class="menu-section">
        <h1 @click="toggleSection('schedule')" class="menu-title">
          <span class="title-text" v-show="!collapsed">调度显示</span>
          <span class="arrow" v-show="!collapsed">{{ openSections.schedule ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.schedule && !collapsed">
          <li><router-link to="/dashboard/schedule">调度页面</router-link></li>
          <li><router-link to="/dashboard/dispatch">调度详情</router-link></li>
        </ul>
      </div>
      
      <div class="menu-section">
        <h1 @click="toggleSection('deepseek')" class="menu-title">
          <span class="title-text" v-show="!collapsed">DeepSeek问答</span>
          <span class="arrow" v-show="!collapsed">{{ openSections.deepseek ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.deepseek && !collapsed">
          <li><router-link to="/dashboard/deepseek">对话界面</router-link></li>
        </ul>
      </div>

      <div class="menu-section">
        <h1 @click="toggleSection('settings')" class="menu-title">
          <span class="title-text" v-show="!collapsed">设置</span>
          <span class="arrow" v-show="!collapsed">{{ openSections.settings ? '▼' : '▶' }}</span>
        </h1>
        <ul v-show="openSections.settings && !collapsed">
          <li><router-link to="/dashboard/settings">系统设置</router-link></li>
          <li><router-link to="/dashboard/profile">个人信息</router-link></li>
        </ul>
      </div>
    </nav>
  </aside>
</template>

<script setup>
import { reactive } from 'vue'

// 接收props
const props = defineProps({
    collapsed: {
        type: Boolean,
        default: false
    }
})

// 定义事件
const emit = defineEmits(['toggle-collapse'])

const openSections = reactive({
  predict: true,  // 默认展开或折叠
  schedule: true,
  settings: true,
  deepseek: true
})

function toggleSection(section) {
  // 收起状态下不允许展开菜单
  if (props.collapsed) return
  openSections[section] = !openSections[section]
}

// 处理切换事件
const handleToggle = () => {
    emit('toggle-collapse')
}
</script>

<style scoped>
.sidebar {
  width: 200px;
  background-color: #001f4d;
  padding: 1rem;
  z-index: 100;
  color: #cce0ff;
  transition: width 0.3s ease;
  position: relative;
  overflow: hidden;
}

.sidebar.collapsed {
  width: 60px;
  padding: 1rem 0.5rem;
}

.toggle-btn {
  position: absolute;
  top: 15px;
  right: 15px;
  width: 30px;
  height: 30px;
  background: #003366;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background 0.3s ease;
  z-index: 10;
}

.toggle-btn:hover {
  background: #004488;
}

.hamburger-icon {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  width: 16px;
  height: 12px;
}

.hamburger-icon span {
  display: block;
  width: 100%;
  height: 2px;
  background-color: #99c2ff;
  transition: all 0.3s ease;
}

.sidebar.collapsed .hamburger-icon span:nth-child(1) {
  transform: none;
}

.sidebar.collapsed .hamburger-icon span:nth-child(2) {
  opacity: 1;
}

.sidebar.collapsed .hamburger-icon span:nth-child(3) {
  transform: none;
}

.sidebar:not(.collapsed) .hamburger-icon span:nth-child(1) {
  transform: rotate(45deg) translate(4px, 4px);
}

.sidebar:not(.collapsed) .hamburger-icon span:nth-child(2) {
  opacity: 0;
}

.sidebar:not(.collapsed) .hamburger-icon span:nth-child(3) {
  transform: rotate(-45deg) translate(4px, -4px);
}

.menu-title {
  font-size: 1rem;
  margin: 1rem 0 0.5rem;
  cursor: pointer;
  user-select: none;
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #99c2ff;
  min-height: 24px;
}

.sidebar.collapsed .menu-title {
  display: none;
}

.title-text {
  flex: 1;
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
  display: block;
  padding: 0.25rem 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
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