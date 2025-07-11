<script setup>
import { ref, onMounted, nextTick, computed } from 'vue'
import { useRouter } from 'vue-router'
import { Zoom } from 'ol/control'
import StationInfo from '../../views/dashboard/stationInfo.vue'
import request from '../../api/axios'
import { useStationMap } from '@/composables/useStationMap'

const mapContainer = ref(null)
const searchQuery = ref('')

// 解构 composable 返回的内容
const {
  stations,
  stationStatusMap,
  mapInstance,
  loading,
  initializeMap,
  updateMapDisplay,
  fetchStationLocations,
  fetchAllStationsStatus,
  handleSearch  // 新增
} = useStationMap()

const welcoming = ref('管理员，欢迎您！')
const currentHour = new Date().getHours()
const selectedHour = ref(currentHour.toString().padStart(2, '0'))
const selectedStation = ref(null)
const showStationInfoDialog = ref(false)
const router = useRouter()

// 固定日期：从 localStorage 获取或取今天
const fixedDate = computed(() => {
  return localStorage.getItem('selectedDate') || new Date().toISOString().split('T')[0]
})

function handleStationClick(station) {
  selectedStation.value = station
  showStationInfoDialog.value = true
  console.log('点击了站点:', station)
}

const handleDialogClose = () => {
  showStationInfoDialog.value = false
  selectedStation.value = null
}

const handleHourChange = async () => {
  const predictTime = `${fixedDate.value}T${selectedHour.value}:00:00Z`
  await fetchAllStationsStatus(predictTime)
  updateMapDisplay()
}

const logout = async () => {
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    router.push('/login')
  }
}

onMounted(async () => {
  try {
    await nextTick()
    initializeMap(mapContainer.value, handleStationClick)

    // 添加自定义缩放控件
    const zoomControl = new Zoom({
      className: 'ol-zoom-custom'
    })
    mapInstance.value.addControl(zoomControl)

    // 加载初始数据
    await fetchStationLocations()
    const predictTime = `${fixedDate.value}T${selectedHour.value}:00:00Z`
    await fetchAllStationsStatus(predictTime)
    updateMapDisplay()
  } catch (error) {
    console.error('组件初始化失败:', error)
  }
})
</script>


<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度</h1>
        <div class="search-container">
          <input 
            type="text" 
            placeholder="搜索站点..." 
            class="search-input"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
          />
          <button class="search-button" @click="handleSearch">搜索</button>
        </div>
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

    <!-- 图例 -->
    <div class="legend">
      <div class="legend-item">
        <img src="/icons/BlueLocationRound.svg" width="24" height="24" alt="少">
        <span>少（0–5）</span>
      </div>
      <div class="legend-item">
        <img src="/icons/YellowLocationRound.svg" width="24" height="24" alt="中">
        <span>中（6–10）</span>
      </div>
      <div class="legend-item">
        <img src="/icons/RedLocationRound.svg" width="24" height="24" alt="多">
        <span>多（11+）</span>
      </div>
    </div>


    <!-- 加载状态 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner">
        <div class="spinner"></div>
        <span>加载中...</span>
      </div>
    </div>

    <!-- Map -->
    <div ref="mapContainer" class="map-container"></div>
    
    <!-- 站点信息弹窗 -->
    <StationInfo
      :show="showStationInfoDialog"
      :station="selectedStation"
      :date="fixedDate"
      :hour="selectedHour"
      @update:show="handleDialogClose"
    />
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

.search-container {
  width: 40%;
  display: flex; 
  min-width: 0;
  gap: 8px;
}

.search-input {
  height: 30px; 
  padding: 4px 8px; 
  border-radius: 4px;
  border: 1px solid #ccc;
  flex: 1;
  min-width: 0; 
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

.search-button {
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.2s;
}

.search-button:hover {
  background-color: #0a1580;
}

.right-time {
  display: flex;
  align-items: center;
  gap: 8px;
}

.right-time label {
  font-size: 14px;
  color: #495057;
  white-space: nowrap;
}


.right-time .fixed-date {
  margin-right: 20px;
  font-weight: bold;
  color: #091275;
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

.legend {
  position: absolute;
  top: 120px;
  right: 20px;
  background-color: rgba(255, 255, 255, 0.9);
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.legend-color {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 2px solid #ffffff;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}

.loading-spinner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #091275;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
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