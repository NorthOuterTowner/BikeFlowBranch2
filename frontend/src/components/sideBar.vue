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
    
    <!-- Logo 区域 -->
    <div class="logo-section" v-show="!collapsed">
      <div class="logo-icon">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="16" cy="16" r="14" fill="url(#gradient)" stroke="#4A90E2" stroke-width="2"/>
          <path d="M12 16L14.5 18.5L20 13" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style="stop-color:#4A90E2;stop-opacity:1" />
              <stop offset="100%" style="stop-color:#357ABD;stop-opacity:1" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div class="logo-text">
        <div class="logo-title">智能调度</div>
        <div class="logo-subtitle">管理平台</div>
      </div>
    </div>
    
    <nav class="navigation" v-show="!collapsed">
      <div class="menu-section">
        <div @click="toggleSection('predict')" class="menu-title">
          <div class="menu-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
              <polyline points="3.27,6.96 12,12.01 20.73,6.96"/>
              <line x1="12" y1="22.08" x2="12" y2="12"/>
            </svg>
          </div>
          <span class="title-text" v-show="!collapsed">预测显示</span>
          <span class="arrow" v-show="!collapsed">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
                 :class="{ 'rotated': openSections.predict }">
              <polyline points="6,9 12,15 18,9"/>
            </svg>
          </span>
        </div>
        <ul v-show="openSections.predict && !collapsed" class="menu-items">
          <li><router-link to="/dashboard/mapView">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
              <circle cx="12" cy="10" r="3"/>
            </svg>
            预测地图
          </router-link></li>
          <li><router-link to="/dashboard/predict">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
            </svg>
            预测结果
          </router-link></li>
        </ul>
      </div>
      
      <div class="menu-section">
        <div @click="toggleSection('schedule')" class="menu-title">
          <div class="menu-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
              <line x1="16" y1="2" x2="16" y2="6"/>
              <line x1="8" y1="2" x2="8" y2="6"/>
              <line x1="3" y1="10" x2="21" y2="10"/>
            </svg>
          </div>
          <span class="title-text" v-show="!collapsed">调度显示</span>
          <span class="arrow" v-show="!collapsed">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
                 :class="{ 'rotated': openSections.schedule }">
              <polyline points="6,9 12,15 18,9"/>
            </svg>
          </span>
        </div>
        <ul v-show="openSections.schedule && !collapsed" class="menu-items">
          <li><router-link to="/dashboard/schedule">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14,2 14,8 20,8"/>
              <line x1="16" y1="13" x2="8" y2="13"/>
              <line x1="16" y1="17" x2="8" y2="17"/>
              <polyline points="10,9 9,9 8,9"/>
            </svg>
            调度页面
          </router-link></li>
          <li><router-link to="/dashboard/dispatch">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
            调度详情
          </router-link></li>
        </ul>
      </div>
      
      <div class="menu-section">
        <div @click="toggleSection('deepseek')" class="menu-title">
          <div class="menu-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              <path d="M13 8H7"/>
              <path d="M17 12H7"/>
            </svg>
          </div>
          <span class="title-text" v-show="!collapsed">DeepSeek问答</span>
          <span class="arrow" v-show="!collapsed">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
                 :class="{ 'rotated': openSections.deepseek }">
              <polyline points="6,9 12,15 18,9"/>
            </svg>
          </span>
        </div>
        <ul v-show="openSections.deepseek && !collapsed" class="menu-items">
          <li><router-link to="/dashboard/deepseek">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M8 12h.01"/>
              <path d="M12 12h.01"/>
              <path d="M16 12h.01"/>
              <path d="M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
            </svg>
            对话界面
          </router-link></li>
        </ul>
      </div>

      <!-- 新增教程部分 -->
      <div class="menu-section">
        <div @click="toggleSection('tutorial')" class="menu-title">
          <div class="menu-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
          </div>
          <span class="title-text" v-show="!collapsed">文档</span>
          <span class="arrow" v-show="!collapsed">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
                 :class="{ 'rotated': openSections.tutorial }">
              <polyline points="6,9 12,15 18,9"/>
            </svg>
          </span>
        </div>
        <ul v-show="openSections.tutorial && !collapsed" class="menu-items">
          <li><router-link to="/dashboard/guide">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M9 11H5a2 2 0 0 0-2 2v3a2 2 0 0 0 2 2h4v-7z"/>
              <path d="M15 11h4a2 2 0 0 1 2 2v3a2 2 0 0 1-2 2h-4v-7z"/>
              <path d="M9 7V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v3"/>
            </svg>
            教程指南
          </router-link></li>
          <li><router-link to="/dashboard/faq">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/>
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
              <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
            常见问题
          </router-link></li>
        </ul>
      </div>

      <div class="menu-section">
        <div @click="toggleSection('statistics')" class="menu-title">
          <div class="menu-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M18 20V10"/>
              <path d="M12 20V4"/>
              <path d="M6 20v-6"/>
            </svg>
          </div>
          <span class="title-text" v-show="!collapsed">数据统计</span>
          <span class="arrow" v-show="!collapsed">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
                 :class="{ 'rotated': openSections.statistics }">
              <polyline points="6,9 12,15 18,9"/>
            </svg>
          </span>
        </div>
        <ul v-show="openSections.statistics && !collapsed" class="menu-items">
          <li><router-link to="/dashboard/statistics">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M3 3v18h18"/>
              <path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
            </svg>
            数据一览
          </router-link></li>
        </ul>
      </div>
      

      <div class="menu-section">
        <div @click="toggleSection('settings')" class="menu-title">
          <div class="menu-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
          </div>
          <span class="title-text" v-show="!collapsed">系统设置</span>
          <span class="arrow" v-show="!collapsed">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" 
                 :class="{ 'rotated': openSections.settings }">
              <polyline points="6,9 12,15 18,9"/>
            </svg>
          </span>
        </div>
        <ul v-show="openSections.settings && !collapsed" class="menu-items">
          <li><router-link to="/dashboard/settings">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>
              <circle cx="12" cy="12" r="3"/>
            </svg>
            系统设置
          </router-link></li>
          <li><router-link to="/dashboard/profile">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
              <circle cx="12" cy="7" r="4"/>
            </svg>
            个人信息
          </router-link></li>
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
  predict: false,  
  schedule: false,
  settings: false,
  deepseek: false,
  statistics: false,
  tutorial: false,  // 新增教程部分的状态
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
  width: 280px;
  background: linear-gradient(180deg, #001a3d 0%, #002856 100%);
  padding: 0;
  z-index: 100;
  color: #e8f4fd;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
  border-right: 1px solid rgba(255, 255, 255, 0.1);
}

.sidebar.collapsed {
  width: 70px;
}

.sidebar.collapsed .toggle-btn {
  right: 17px;
}

.toggle-btn {
  position: absolute;
  top: 20px;
  right: 20px;
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  z-index: 10;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.toggle-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  background: linear-gradient(135deg, #5BA0F2 0%, #4A8ACD 100%);
}

.hamburger-icon {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  width: 18px;
  height: 14px;
}

.hamburger-icon span {
  display: block;
  width: 100%;
  height: 2px;
  background: white;
  border-radius: 1px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
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
  transform: rotate(45deg) translate(5px, 5px);
}

.sidebar:not(.collapsed) .hamburger-icon span:nth-child(2) {
  opacity: 0;
}

.sidebar:not(.collapsed) .hamburger-icon span:nth-child(3) {
  transform: rotate(-45deg) translate(5px, -5px);
}

.logo-section {
  display: flex;
  align-items: center;
  padding: 24px 20px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 20px;
}

.logo-icon {
  min-width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
}

.sidebar.collapsed .logo-section {
  justify-content: center;
  padding: 24px 15px;
}

.logo-text {
  margin-left: 12px;
  opacity: 1;
  transition: opacity 0.3s ease;
}

.logo-title {
  font-size: 18px;
  font-weight: 700;
  color: #ffffff;
  line-height: 1.2;
  margin: 0;
}

.logo-subtitle {
  font-size: 12px;
  color: #94c7ff;
  margin-top: 2px;
  opacity: 0.8;
}

.navigation {
  padding: 0 12px;
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  scrollbar-width: thin;
  scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
}

.navigation::-webkit-scrollbar {
  width: 4px;
}

.navigation::-webkit-scrollbar-track {
  background: transparent;
}

.navigation::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 2px;
}

.menu-section {
  margin-bottom: 8px;
}

.menu-title {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  color: #94c7ff;
  border-radius: 8px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.menu-title::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, rgba(74, 144, 226, 0.1) 0%, rgba(53, 122, 189, 0.1) 100%);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.menu-title:hover::before {
  opacity: 1;
}

.menu-title:hover {
  color: #ffffff;
  transform: translateX(4px);
}

.sidebar.collapsed .menu-title {
  justify-content: center;
  padding: 12px 8px;
  margin-bottom: 8px;
}

.menu-icon {
  min-width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 12px;
  color: #4A90E2;
  z-index: 1;
}

.sidebar.collapsed .menu-icon {
  margin-right: 0;
}

.sidebar.collapsed .menu-title:hover {
  transform: none;
}

.sidebar.collapsed .menu-items {
  display: none;
}

.title-text {
  flex: 1;
  font-size: 15px;
  font-weight: 500;
  z-index: 1;
}

.arrow {
  margin-left: 8px;
  transition: transform 0.3s ease;
  z-index: 1;
}

.arrow.rotated {
  transform: rotate(180deg);
}

.menu-items {
  list-style: none;
  padding: 0;
  margin: 0 0 0 36px;
  overflow: hidden;
}

.menu-items li {
  margin-bottom: 4px;
}

.menu-items a {
  display: flex;
  align-items: center;
  color: #b8d4f1;
  text-decoration: none;
  padding: 10px 16px;
  border-radius: 6px;
  transition: all 0.3s ease;
  font-size: 14px;
  position: relative;
  overflow: hidden;
}

.menu-items a::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(90deg, rgba(74, 144, 226, 0.15) 0%, rgba(53, 122, 189, 0.15) 100%);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.menu-items a:hover::before {
  opacity: 1;
}

.menu-items a:hover {
  color: #ffffff;
  transform: translateX(6px);
}

.menu-items a.router-link-exact-active {
  color: #ffffff;
  background: linear-gradient(90deg, rgba(74, 144, 226, 0.3) 0%, rgba(53, 122, 189, 0.3) 100%);
  border-left: 3px solid #4A90E2;
  font-weight: 600;
}

.menu-items a svg {
  width: 16px;
  height: 16px;
  margin-right: 12px;
  opacity: 0.8;
  z-index: 1;
}

.menu-items a:hover svg,
.menu-items a.router-link-exact-active svg {
  opacity: 1;
}

/* 动画效果 */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.menu-items {
  animation: slideIn 0.3s ease;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    width: 100%;
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    z-index: 1000;
  }
  
  .sidebar.collapsed {
    width: 70px;
  }
}
</style>