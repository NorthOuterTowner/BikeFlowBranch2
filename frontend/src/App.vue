<template>
  <n-config-provider :theme-overrides="themeOverrides">
    <router-view />
  </n-config-provider>
</template>

<script setup>
  import { NConfigProvider } from 'naive-ui'

  const themeOverrides = {
    common: {
      primaryColor: '#0556a7',
      primaryColorHover: '#0b5fb3',
      primaryColorPressed: '#044c8c',
      primaryColorSuppl: '#0556a7'
    }
  }
</script>

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