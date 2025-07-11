<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import { fromLonLat } from 'ol/proj'

const mapContainer = ref(null)
let mapInstance = null
const router = useRouter()

const welcoming = ref('管理员，欢迎您！')
const fixedDate = computed(() => {
  return localStorage.getItem('selectedDate') || new Date().toISOString().split('T')[0]
})
const currentHour = getCurrentHourString()

function getCurrentHourString() {
  const now = new Date()
  const hour = now.getHours().toString().padStart(2, '0')
  return `${hour}:00`
}

/**
 * 登出功能
 */
const logout = async () => {
  router.push('/login')
}

/**
 * 初始化地图
 */
onMounted(async () => {
  mapInstance = new Map({
    target: mapContainer.value,
    layers: [new TileLayer({ source: new OSM() })],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]),
      zoom: 11,
      maxZoom: 20,
      minZoom: 3
    }),
    controls: [] 

  })

  const zoomControl = new Zoom({
    className: 'ol-zoom-custom'
  })
  mapInstance.addControl(zoomControl)

})
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度</h1>
      </div>
      <div class="user-info">
        <div class="user-top">
          <span class="welcoming">{{ welcoming }}</span>
          <button class="logout-button" @click="logout">退出</button>
        </div>

        <div class="right-time">
          <label>日期：</label>
          <span class="fixed-date">{{ fixedDate }}</span>
          <label>选择时段：</label>
          <span class="fixed-time">{{ currentHour }}</span>
        </div>
      </div>
    </header>

    <!-- Map -->
    <div ref="mapContainer" class="map-container"></div>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  position: relative;
}

.app-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 10px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ccc;
  flex-shrink: 0;
  width: 100%;
  z-index: 50;
  box-sizing: border-box;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
  min-width: 0; 
}

.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
}

.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-left: 20px;
  gap: 15px;
  flex-shrink: 0;
}

.user-top {
  display: flex;
  align-items: center;
  gap: 20px;
}

.welcoming {
  font-size: 14px;
  white-space: nowrap;
  color: #495057;
}

.logout-button{
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.2s;
}

.logout-button:hover {
  background-color: #0a1580;
}

.fixed-date {
  margin-right: 20px;
  font-weight: bold;
}
.fixed-time {
  margin-right: 20px;
  font-weight: bold;
}

.disabled-option {
  color: #999;
  background-color: #f2f2f2;
}

.map-container {
  flex: 1;
  width: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  position: relative;
  z-index: 1;
}

.map-container :deep(.ol-zoom-custom) {
  position: absolute;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
}

.map-container :deep(.ol-zoom-custom button) {
  width: 60px;
  height: 60px;
  font-size: 24px;
  background-color: rgba(255, 255, 255, 0.95);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  transition: background-color 0.2s;
}

.map-container :deep(.ol-zoom-custom button:hover) {
  background-color: #f0f0f0;
}

.map-container :deep(.ol-attribution) {
  position: absolute;
  bottom: 0;
  right: 0;
  max-width: calc(100% - 1.3em);
  display: flex;
  flex-flow: row-reverse;
  align-items: center;
}

.map-container :deep(.ol-attribution ul) {
  margin: 0;
  padding: 1px 0.5em;
  color: #000;
  text-shadow: 0 1px 0 rgba(255,255,255,.9);
  font-size: 12px;
}

.map-container :deep(.ol-attribution button) {
  flex-shrink: 0;
  color: #000;
  background-color: rgba(255,255,255,.5);
  border: none;
  outline: none;
  cursor: pointer;
  padding: 2px;
  margin: 2px;
  border-radius: 2px;
}
</style>