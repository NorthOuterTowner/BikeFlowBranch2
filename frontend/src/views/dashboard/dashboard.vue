<template>
    <div class="dashboard-layout">
        <SideBar 
            :collapsed="isCollapsed" 
            @toggle-collapse="toggleSidebar"
        />  
        <div class="content" :class="{ 'content-expanded': isCollapsed }"> 
            <router-view />     <!-- 右侧显示当前子页面 -->
        </div>
    </div>
</template>
  
<script setup>
import { ref } from 'vue'
import SideBar from '../../components/sideBar.vue'

// 控制边栏收起状态
const isCollapsed = ref(false)

// 切换边栏状态
const toggleSidebar = () => {
    isCollapsed.value = !isCollapsed.value
}
</script>
  
<style scoped>
.dashboard-layout {
    display: flex;
    min-height: 100vh;
    width: 100vw; 
    margin: 0; 
    padding: 0; 
}

.content {
    flex: 1;
    height: 100vh; 
    background: #fff;
    overflow: hidden;
    margin: 0;
    padding: 0;
    transition: margin-left 0.3s ease;
}

/* 当边栏收起时，内容区域可以获得更多空间 */
.content-expanded {
    margin-left: 0;
}
</style>