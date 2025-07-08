<template>
  <router-view />   <!-- 根据路由切换页面：先显示 Login.vue，登录后显示 Dashboard.vue -->
</template>


<style>
/* 全局样式重置 */
/* 在你的全局CSS文件中添加这些样式，或者在App.vue的<style>标签中添加 */

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0;
  overflow-x: hidden; /* 防止水平滚动条 */
}

#app {
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0;
}

/* 如果你使用了Vue Router，也重置router-view */
.router-view {
  width: 100%;
  height: 100%;
}
</style>