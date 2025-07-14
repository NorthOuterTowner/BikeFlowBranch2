<script setup>
import { ref, onMounted, computed } from 'vue'
import request from '../../api/axios'
import { useRouter } from 'vue-router'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import { fromLonLat } from 'ol/proj'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import VectorSource from 'ol/source/Vector'
import { Heatmap as HeatmapLayer } from 'ol/layer'
import {Zoom } from 'ol/control'

const mapContainer = ref(null)
let mapInstance = null
let heatmapLayer = null
const router = useRouter()

// 状态管理
const stations = ref([])
const predictionData = ref([])
const loading = ref(false)
const welcoming = ref('管理员，欢迎您！')

// 表格相关状态
const showTable = ref(true)

// 从localStorage获取日期，如果没有则使用当前日期
const selectedDate = computed(() => {
  return localStorage.getItem('selectedDate') || new Date().toISOString().split('T')[0]
})

const currentHour = new Date().getHours()
const selectedHour = ref(currentHour.toString().padStart(2, '0'))

// 差值统计
const differenceStats = ref({
  maxInflow: 0,
  maxOutflow: 0,
  inflowStations: 0,
  outflowStations: 0,
  totalStations: 0
})

// 合并站点和预测数据
const combinedData = computed(() => {
  const predictionMap = new Map()
  predictionData.value.forEach(item => {
    predictionMap.set(item.station_id, item)
  })

  return stations.value.map(station => {
    const prediction = predictionMap.get(station.station_id)
    return {
      ...station,
      ...(prediction || { inflow: 0, outflow: 0, stock: 0 }),
      difference: prediction ? prediction.inflow - prediction.outflow : 0
    }
  })
})

async function fetchStationLocations() {
  console.log('获取站点位置...')
  try {
    loading.value = true
    const response = await request.get('/stations/locations')
    
    const data = response.data
    if (Array.isArray(data)) {
      stations.value = data
    } else if (data && Array.isArray(data.data)) {
      stations.value = data.data
    } else {
      console.error('站点数据格式不正确:', data)
      stations.value = []
    }
    
    console.log('获取到站点数据:', stations.value)
    return stations.value
  } catch (error) {
    console.error('获取站点位置失败:', error)
    stations.value = []
    return []
  } finally {
    loading.value = false
  }
}

async function fetchPredictionData(predictTime) {
  try {
    loading.value = true
    const response = await request.get('/predict/stations/all', {
      params: { predict_time: predictTime }
    })
    
    predictionData.value = response.data.stations_status || []
    console.log('获取预测数据:', predictionData.value)
    updateHeatmapWithBlur()
  } catch (error) {
    console.error('获取预测数据失败:', error)
    predictionData.value = []
  } finally {
    loading.value = false
  }
}

function calculateDifferenceStats() {
  const predictionMap = new Map()
  predictionData.value.forEach(item => {
    predictionMap.set(item.station_id, item)
  })

  let maxInflow = 0
  let maxOutflow = 0
  let inflowCount = 0
  let outflowCount = 0
  let totalCount = 0

  stations.value.forEach(station => {
    const prediction = predictionMap.get(station.station_id)
    if (prediction) {
      const difference = prediction.inflow - prediction.outflow
      totalCount++
      
      if (difference > 0) {
        inflowCount++
        maxInflow = Math.max(maxInflow, difference)
      } else if (difference < 0) {
        outflowCount++
        maxOutflow = Math.max(maxOutflow, Math.abs(difference))
      }
    }
  })

  differenceStats.value = {
    maxInflow,
    maxOutflow,
    inflowStations: inflowCount,
    outflowStations: outflowCount,
    totalStations: totalCount
  }
}

function updateHeatmapWithBlur() {
  if (!mapInstance || !stations.value.length || !predictionData.value.length) {
    console.warn('地图未初始化或数据不完整')
    return
  }

  calculateDifferenceStats()

  const predictionMap = new Map()
  predictionData.value.forEach(item => {
    predictionMap.set(item.station_id, item)
  })

  // 清理旧图层
  if (heatmapLayer) {
    mapInstance.removeLayer(heatmapLayer)
  }

  const heatmapSource = new VectorSource()

  const maxAbsDifference = Math.max(differenceStats.value.maxInflow, differenceStats.value.maxOutflow)

  stations.value.forEach(station => {
    const prediction = predictionMap.get(station.station_id)
    if (!prediction) return

    const difference = prediction.inflow - prediction.outflow
    const normalized = difference / maxAbsDifference // -1 到 1

    const feature = new Feature({
      geometry: new Point(fromLonLat([
        parseFloat(station.longitude),
        parseFloat(station.latitude)
      ]))
    })

    // 映射到 0~1，0 代表最大流出，0.5 代表中性，1 代表最大流入
  const heatmapWeight = Math.pow((normalized + 1) / 2, 0.5)  // 开根号，放大低权重区颜色


    feature.set('weight', heatmapWeight)
    feature.set('stationData', { ...station, ...prediction })
    stations.value.forEach(station => {
  const prediction = predictionMap.get(station.station_id)
  if (!prediction) return

  const difference = prediction.inflow - prediction.outflow
  const normalized = difference / maxAbsDifference
  const weight = (normalized + 1) / 2

  console.log(`站点 ${station.station_id}：差值=${difference}, 归一化=${normalized.toFixed(2)}, 权重=${weight.toFixed(2)}`)
})

    heatmapSource.addFeature(feature)
  })

  heatmapLayer = new HeatmapLayer({
    source: heatmapSource,
    blur: 50,
    radius: 50,
    weight: feature => feature.get('weight') || 0,
    gradient: [
      'rgba(0, 0, 100, 1)',      // 更深蓝黑
      'rgba(0, 0, 139, 1)',      // 深蓝
      'rgba(0, 0, 180, 1)',      // 深蓝偏亮
      'rgba(0, 0, 220, 1)',      // 更亮蓝
      'rgba(0, 0, 255, 1)',      // 纯蓝
      'rgba(0, 64, 255, 1)',     // 蓝带点青
      'rgba(0, 128, 255, 1)',    // 蓝青
      'rgba(0, 192, 255, 1)',    // 浅蓝青
      'rgba(0, 255, 255, 1)',    // 青色
      'rgba(0, 255, 128, 1)',    // 青绿
      'rgba(0, 255, 0, 1)',      // 绿色（最显眼）
      'rgba(173, 255, 47, 1)',   // 黄绿色
      'rgba(255, 255, 0, 1)',    // 黄色（中性）
      'rgba(255, 165, 0, 1)',    // 橙色
      'rgba(255, 0, 0, 0.7)',      // 红色（最大）
    ]



  })

  heatmapLayer.setZIndex(2)
  mapInstance.addLayer(heatmapLayer)
}


const handleTimeChange = async () => {
  const predictTime = `${selectedDate.value}T${selectedHour.value}:00:00Z`
  await fetchPredictionData(predictTime)
}

const toggleTable = () => {
  showTable.value = !showTable.value
}

const logout = async () => {
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    // 清除所有 sessionStorage 项
    sessionStorage.clear()
    router.push('/login')
  }
}

onMounted(async () => {
  // 获取站点位置
  await fetchStationLocations()

  // 初始化地图
  mapInstance = new Map({
    target: mapContainer.value,
    layers: [new TileLayer({ source: new OSM() })],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]),
      zoom: 14,
      maxZoom: 20,
      minZoom: 3
    }),
    controls: [] 

  })

  const zoomControl = new Zoom({
  className: 'ol-zoom-custom'
})
mapInstance.addControl(zoomControl)
  // 加载初始数据
  const predictTime = `${selectedDate.value}T${selectedHour.value}:00:00Z`
  await fetchPredictionData(predictTime)
})

</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车流量差值热力图</h1>
        <div class="search-container">
          <button class="toggle-table-button" @click="toggleTable">
            {{ showTable ? '隐藏表格' : '显示表格' }}
          </button>
        </div>
      </div>
      <div class="user-info">
        <div class="user-top">
          <span class="welcoming">{{ welcoming }}</span>
          <button class="logout-button" @click="logout">退出</button>
        </div>

        <div class="control-panel">
          <div class="time-control">
            <label>日期：</label>
            <span class="selected-date">{{ selectedDate }}</span>
            <label>选择时段：</label>
            <select v-model="selectedHour" @change="handleTimeChange">
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
      </div>
    </header>

    <!-- 主要内容区域 -->
    <div class="main-content">
      <!-- 站点预测表格 -->
      <div v-if="showTable" class="table-panel">
        <div class="table-header">
          <h3>站点预测结果</h3>
          <div class="table-controls">
            <span class="table-count">共 {{ combinedData.length }} 个站点</span>
          </div>
        </div>
        
        <div class="table-container">
          <table class="stations-table">
            <thead>
              <tr>
                <th>站点ID</th>
                <th>站点名称</th>
                <th>入车流</th>
                <th>出车流</th>
                <th>差值</th>
                <th>库存</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="station in combinedData" :key="station.station_id">
                <td class="station-id">{{ station.station_id }}</td>
                <td class="station-name">{{ station.station_name }}</td>
                <td class="inflow">{{ station.inflow }}</td>
                <td class="outflow">{{ station.outflow }}</td>
                <td class="difference" :class="{
                  'positive': station.difference > 0,
                  'negative': station.difference < 0,
                  'zero': station.difference === 0
                }">
                  {{ station.difference > 0 ? '+' : '' }}{{ station.difference }}
                </td>
                <td class="stock">{{ station.stock }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- 地图区域 -->
      <div class="map-section">
        <!-- 流量差值统计面板 -->
        <div class="stats-panel">
          <div class="stats-title">流量差值统计</div>
          <div class="stats-grid">
            <div class="stat-item inflow">
              <div class="stat-value">{{ differenceStats.inflowStations }}</div>
              <div class="stat-label">汇入站点</div>
            </div>
            <div class="stat-item outflow">
              <div class="stat-value">{{ differenceStats.outflowStations }}</div>
              <div class="stat-label">汇出站点</div>
            </div>
            <div class="stat-item max-inflow">
              <div class="stat-value">+{{ differenceStats.maxInflow }}</div>
              <div class="stat-label">最大汇入</div>
            </div>
            <div class="stat-item max-outflow">
              <div class="stat-value">-{{ differenceStats.maxOutflow }}</div>
              <div class="stat-label">最大汇出</div>
            </div>
          </div>
        </div>

        <!-- 热力图图例 -->
        <div class="heatmap-legend">
          <div class="legend-title">流量差值 (入车流 - 出车流)</div>
          <div class="legend-gradient">
            <div class="gradient-bar"></div>
            <div class="legend-labels">
              <span class="outflow-label">汇出</span>
              <span class="balance-label">平衡</span>
              <span class="inflow-label">汇入</span>
            </div>
          </div>
        </div>

        <!-- 地图容器 -->
        <div ref="mapContainer" class="map-container"></div>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner">
        <div class="spinner"></div>
        <span>加载预测数据中...</span>
      </div>
    </div>
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
  color: #010210;
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

.logout-button, .toggle-table-button {
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.2s;
}

.logout-button:hover, .toggle-table-button:hover {
  background-color: #0a1580;
}

.control-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: flex-end;
}

.time-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.selected-date {
  font-weight: bold;
  color: #091275;
  margin-right: 15px;
}

.time-control select {
  padding: 6px 10px;
  font-size: 14px;
  height: 30px;
  border-radius: 4px;
  border: 1px solid #ccc;
  background-color: white;
}

.time-control label {
  font-size: 14px;
  color: #495057;
  white-space: nowrap;
}

.main-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.table-panel {
  width: 400px;
  background-color: white;
  border-right: 1px solid #ccc;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.table-header {
  padding: 15px;
  background-color: #f8f9fa;
  border-bottom: 1px solid #dee2e6;
  flex-shrink: 0;
}

.table-header h3 {
  margin: 0 0 10px 0;
  color: #091275;
  font-size: 16px;
}

.table-controls {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.table-count {
  font-size: 12px;
  color: #666;
  text-align: right;
}

.table-container {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.stations-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.stations-table th {
  background-color: #f8f9fa;
  padding: 8px 6px;
  text-align: left;
  border-bottom: 2px solid #dee2e6;
  font-weight: bold;
  color: #091275;
  position: sticky;
  top: 0;
  z-index: 10;
}

.stations-table td {
  padding: 8px 6px;
  border-bottom: 1px solid #dee2e6;
  vertical-align: middle;
}

.stations-table tr:hover {
  background-color: #f8f9fa;
}

.station-id {
  font-weight: bold;
  color: #091275;
}

.station-name {
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.inflow {
  color: #0066ff;
  font-weight: bold;
}

.outflow {
  color: #ff6600;
  font-weight: bold;
}

.difference.positive {
  color: #00cc00;
  font-weight: bold;
}

.difference.negative {
  color: #cc0000;
  font-weight: bold;
}

.difference.zero {
  color: #666;
}

.stock {
  color: #333;
  font-weight: bold;
}

.map-section {
  flex: 1;
  position: relative;
  overflow: hidden;
}

.map-container {
  width: 100%;
  height: 100%;
  position: relative;
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

.stats-panel {
  position: absolute;
  top: 20px;
  left: 20px;
  background-color: rgba(255, 255, 255, 0.95);
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  z-index: 10;
  min-width: 200px;
}

.stats-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 15px;
  color: #091275;
  text-align: center;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 15px;
}

.stat-item {
  text-align: center;
  padding: 10px;
  border-radius: 6px;
  background-color: rgba(255, 255, 255, 0.8);
  border: 2px solid;
}

.stat-item.inflow {
  border-color: #0066ff;
  background-color: rgba(0, 102, 255, 0.1);
}

.stat-item.outflow {
  border-color: #ff6600;
  background-color: rgba(255, 102, 0, 0.1);
}

.stat-item.max-inflow {
  border-color: #00cc00;
  background-color: rgba(0, 204, 0, 0.1);
}

.stat-item.max-outflow {
  border-color: #cc0000;
  background-color: rgba(204, 0, 0, 0.1);
}

.stat-value {
  font-size: 20px;
  font-weight: bold;
  color: #091275;
  margin-bottom: 5px;
}

.stat-label {
  font-size: 12px;
  color: #666;
}

.heatmap-legend {
  position: absolute;
  top: 20px;
  right: 20px;
  background-color: rgba(255, 255, 255, 0.95);
  padding: 15px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
  z-index: 10;
  min-width: 140px;
}

.legend-title {
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 10px;
  color: #091275;
  text-align: center;
}

.legend-gradient {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.gradient-bar {
  height: 20px;
  background: linear-gradient(to right, 
    rgba(0, 100, 255, 1) 0%,      /* 蓝色 - 最大流出 */
    rgba(0, 150, 255, 0.8) 25%,   /* 深蓝色 */
    rgba(255, 255, 0, 0.8) 50%,   /* 黄色 - 平衡 */
    rgba(255, 100, 0, 0.9) 75%,   /* 橙色 */
    rgba(255, 0, 0, 1) 100%       /* 红色 - 最大流入 */
  );
  border-radius: 10px;
  border: 1px solid #ddd;
}

.legend-labels {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-top: 5px;
}

.outflow-label {
  color: #0066ff;
  font-weight: bold;
}

.balance-label {
  color: #ff9900;
  font-weight: bold;
}

.inflow-label {
  color: #ff0000;
  font-weight: bold;
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

/* 响应式设计 */
@media (max-width: 1200px) {
  .table-panel {
    width: 350px;
  }
  
  .stations-table {
    font-size: 11px;
  }
  
  .stations-table th,
  .stations-table td {
    padding: 6px 4px;
  }
}

@media (max-width: 768px) {
  .app-header {
    flex-direction: column;
    gap: 15px;
  }
  
  .header-left {
    width: 100%;
  }
  
  .search-container {
    width: 100%;
  }
  

  
  .control-panel {
    align-items: stretch;
  }
  
  .time-control {
    justify-content: space-between;
  }
  
  .main-content {
    flex-direction: column;
  }
  
  .table-panel {
    width: 100%;
    height: 300px;
    border-right: none;
    border-bottom: 1px solid #ccc;
  }
  
  .map-section {
    flex: 1;
  }
  
  .stats-panel {
    top: 10px;
    left: 10px;
    right: 10px;
    min-width: auto;
  }
  
  .stats-grid {
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }
  
  .heatmap-legend {
    top: 10px;
    right: 10px;
    left: 10px;
  }
}
</style>