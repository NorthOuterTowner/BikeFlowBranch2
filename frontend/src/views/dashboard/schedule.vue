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
const currentHour = new Date().getHours()
const selectedHour = ref(currentHour.toString().padStart(2, '0'))

const handleHourChange = async () => {
  console.log('选择的时间:', `${fixedDate.value}T${selectedHour.value}:00:00Z`)
  // 可以在这里添加处理逻辑
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
          <select v-model="selectedHour" @change="handleHourChange">
            <option
              v-for="h in 24"
              :key="h"
              :value="(h - 1).toString().padStart(2, '0')"
              :disabled="(h - 1) < currentHour"
              :class="{ 'disabled-option': (h - 1) < currentHour }"
            >
              {{ (h - 1).toString().padStart(2, '0') }}:00
            </option>
          </select>
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

.right-time {
  display: flex;
  align-items: center;
  gap: 8px;
}

.right-time .fixed-date {
  margin-right: 20px;
  font-weight: bold;
}

.right-time select {
  padding: 6px 10px;
  font-size: 14px;
  height: 30px;
  border-radius: 4px;
  border: 1px solid #ccc;
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

/* OpenLayers 样式覆盖 */
.map-container :deep(.ol-zoom) {
  position: absolute;
  top: 0.5em;
  left: 0.5em;
  background: rgba(255,255,255,.4);
  border-radius: 4px;
  padding: 2px;
}

.map-container :deep(.ol-zoom button) {
  display: block;
  margin: 1px;
  padding: 0;
  color: white;
  font-size: 1.14em;
  font-weight: 700;
  text-decoration: none;
  text-align: center;
  height: 1.375em;
  width: 1.375em;
  background-color: rgba(0,60,136,.5);
  border: none;
  border-radius: 2px;
  cursor: pointer;
}

.map-container :deep(.ol-zoom button:hover) {
  background-color: rgba(0,60,136,.7);
}

.map-container :deep(.ol-zoom button:focus) {
  background-color: rgba(0,60,136,.7);
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