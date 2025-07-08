<template>
    <div class="dashboard-layout">
        <SideBar />  
        <div class="content"> 
            <router-view />     <!-- 右侧显示当前子页面 -->
        </div>
    </div>
</template>
  
<script setup>
import SideBar from '../../components/SideBar.vue'
</script>
  
<style scoped>
.dashboard-layout {
    display: flex;
    min-height: 100vh;   /* 高度铺满整个视口 */
}
.content {
    flex: 1;
    padding: 16px;
    height: 100vh; 
    background: #fff;    /* 背景颜色 */
    overflow: auto; 
}
</style>
  