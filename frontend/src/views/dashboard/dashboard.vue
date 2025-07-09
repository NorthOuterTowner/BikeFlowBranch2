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
}
</style>