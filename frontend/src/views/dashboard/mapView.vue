<script setup>
import { ref, onMounted } from 'vue'

const mapContainer = ref(null)
let mapInstance = null

onMounted(() => {
  if (!window.AMap) {
    console.error('AMap SDK 未加载')
    return
  }

  // 创建地图实例，传入容器DOM和配置项
  mapInstance = new window.AMap.Map(mapContainer.value, {
    zoom: 11,                   // 缩放级别
    center: [116.397428, 39.90923],  // 中心点
    viewMode: '3D',             // 3D模式
  })
})
</script>

<template>
  <div class="map-view">
    <div 
      ref="mapContainer" 
      style="width: 100%; height: 100%; border: 1px solid #ccc;"
    ></div>
  </div>
</template>

<style scoped>
.map-view {
  width: 100%;
  height: 100%;
}
</style>