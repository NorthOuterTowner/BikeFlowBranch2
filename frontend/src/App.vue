<script setup>
import Cesium from './views/dashboard/mapView.vue'
</script>

<template>
  <div id="app" class="full-screen">
    <Cesium />
  </div>
</template>

<style>
/* 全局样式重置 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  height: 100%;
  width: 100%;
  overflow: hidden;
}

#app {
  height: 100vh;
  width: 100vw;
  position: relative;
}

.full-screen {
  height: 100%;
  width: 100%;
}
</style>