<script setup>
import { ref, onMounted, watch } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import { fromLonLat } from 'ol/proj'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Style from 'ol/style/Style'
import Circle from 'ol/style/Circle'
import Fill from 'ol/style/Fill'
import Stroke from 'ol/style/Stroke'
import Text from 'ol/style/Text'

const mapContainer = ref(null)
let mapInstance = null
let vectorLayer = null
const router = useRouter()
const stations = ref([])
const stationBikeCounts = ref(new Map()) // 存储站点单车数量
const loading = ref(false)

// 根据车辆数量获取颜色（深色表示车辆多，浅色表示车辆少）
function getColorByBikeCount(count) {
  if (count >= 50) {
    return '#1a5490' // 深蓝色 - 车辆多
  } else if (count >= 20) {
    return '#4a90e2' // 中等蓝色 - 车辆中等
  } else {
    return '#87ceeb' // 浅蓝色 - 车辆少
  }
}

// 获取站点样式
function getStationStyle(station) {
  const bikeCount = stationBikeCounts.value.get(station.station_id) || 0
  const color = getColorByBikeCount(bikeCount)
  
  return new Style({
    image: new Circle({
      radius: 8,
      fill: new Fill({
        color: color
      }),
      stroke: new Stroke({
        color: '#ffffff',
        width: 2
      })
    }),
    text: new Text({
      text: bikeCount.toString(),
      fill: new Fill({
        color: '#ffffff'
      }),
      font: '12px Arial',
      textAlign: 'center',
      textBaseline: 'middle'
    })
  })
}

// 获取站点位置数据
async function fetchStationLocations() {
  try {
    const response = await axios.get('/stations/locations')
    stations.value = response.data
  } catch (error) {
    console.error('获取站点位置失败:', error)
  }
}

// 获取单个站点的单车数量
async function fetchStationBikeNum(stationId, date, hour) {
  try {
    const response = await axios.get('/stations/bikeNum', {
      params: {
        station_id: stationId,
        date: date,
        hour: hour
      }
    })
    return response.data.bikeNum || 0
  } catch (error) {
    console.error(`获取站点 ${stationId} 单车数量失败:`, error)
    return 0
  }
}

// 获取所有站点的单车数量
async function fetchAllStationsBikeNum(date, hour) {
  if (!stations.value.length) return

  loading.value = true
  const bikeCounts = new Map()
  
  try {
    // 批量获取所有站点的单车数量
    const promises = stations.value.map(async (station) => {
      const bikeNum = await fetchStationBikeNum(station.station_id, date, hour)
      bikeCounts.set(station.station_id, bikeNum)
    })
    
    await Promise.all(promises)
    stationBikeCounts.value = bikeCounts
    
    // 更新地图显示
    updateMapDisplay()
  } catch (error) {
    console.error('获取站点单车数量失败:', error)
  } finally {
    loading.value = false
  }
}

// 更新地图显示
function updateMapDisplay() {
  if (!mapInstance || !vectorLayer) return
  
  // 清除现有的features
  vectorLayer.getSource().clear()
  
  // 重新创建features
  const features = stations.value.map(station => {
    const feature = new Feature({
      geometry: new Point(fromLonLat([station.longitude, station.latitude]))
    })
    
    // 设置站点样式
    feature.setStyle(getStationStyle(station))
    
    // 将站点数据存储到feature中，便于后续使用
    feature.set('stationData', station)
    
    return feature
  })
  
  // 添加新的features
  vectorLayer.getSource().addFeatures(features)
}

// 解析datetime-local值获取日期和小时
function parseDateTimeLocal(dateTimeString) {
  const date = new Date(dateTimeString)
  const dateStr = date.toISOString().split('T')[0] // YYYY-MM-DD
  const hour = date.getHours().toString()
  return { dateStr, hour }
}

function getNowDatetimeLocal() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hours = String(now.getHours()).padStart(2, '0')
  const minutes = String(now.getMinutes()).padStart(2, '0')

  return `${year}-${month}-${day}T${hours}:${minutes}`
}

const dateTime = ref(getNowDatetimeLocal())
const welcoming = ref('管理员，欢迎您！')
const searchQuery = ref('')

// 监听时间变化，重新获取数据
watch(dateTime, async (newDateTime) => {
  const { dateStr, hour } = parseDateTimeLocal(newDateTime)
  await fetchAllStationsBikeNum(dateStr, hour)
}, { immediate: false })

onMounted(async () => {
  // 先获取站点位置数据
  await fetchStationLocations()
  
  // 初始化地图
  mapInstance = new Map({
    target: mapContainer.value,
    layers: [
      new TileLayer({
        source: new OSM(),
      }),
    ],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]), // 调整到合适的中心点
      zoom: 11,
      maxZoom: 20,
      minZoom: 3
    }),
  })

  // 创建矢量图层
  vectorLayer = new VectorLayer({
    source: new VectorSource()
  })

  mapInstance.addLayer(vectorLayer)
  
  // 获取初始时间的数据
  const { dateStr, hour } = parseDateTimeLocal(dateTime.value)
  await fetchAllStationsBikeNum(dateStr, hour)
})

const logout = async () => {
  try {
    await axios.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  }
  router.push('/login')
}

const handleSearch = () => {
  if (searchQuery.value.trim()) {
    console.log('搜索:', searchQuery.value)
    // 在这里添加搜索逻辑
    // 例如：在地图上高亮显示匹配的站点
    const matchedStations = stations.value.filter(station => 
      station.station_name.toLowerCase().includes(searchQuery.value.toLowerCase()) ||
      station.station_id.toLowerCase().includes(searchQuery.value.toLowerCase())
    )
    
    if (matchedStations.length > 0) {
      // 缩放到第一个匹配的站点
      const station = matchedStations[0]
      mapInstance.getView().animate({
        center: fromLonLat([station.longitude, station.latitude]),
        zoom: 15,
        duration: 1000
      })
    }
  }
}

// 手动刷新数据
const refreshData = async () => {
  const { dateStr, hour } = parseDateTimeLocal(dateTime.value)
  await fetchAllStationsBikeNum(dateStr, hour)
}

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
        <div class="datetime-picker">
          <input 
            type="datetime-local" 
            v-model="dateTime" 
            class="datetime-input"
            placeholder="请选择日期时间"
          />
          <button class="refresh-button" @click="refreshData" :disabled="loading">
            {{ loading ? '加载中...' : '刷新' }}
          </button>
        </div>
      </div>
    </header>

    <!-- 图例 -->
    <div class="legend">
      <div class="legend-item">
        <div class="legend-color" style="background-color: #87ceeb;"></div>
        <span>少 (0-20)</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #4a90e2;"></div>
        <span>中等 (20-50)</span>
      </div>
      <div class="legend-item">
        <div class="legend-color" style="background-color: #1a5490;"></div>
        <span>多 (50+)</span>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner">加载中...</div>
    </div>

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
  align-items: flex-start;
  margin-left: 20px;
  gap: 8px;
  flex-shrink: 0; 
}

.user-top {
  display: flex;
  align-items: center;
  gap: 40px;
}

.welcoming {
  font-size: 14px;
  white-space: nowrap; 
}

.datetime-picker {
  display: flex;
  gap: 8px;
  align-items: center;
}

.datetime-input {
  width: 200px;
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid #ccc;
  cursor: pointer;
  box-sizing: border-box;
}

.logout-button, .search-button, .refresh-button {
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
}

.refresh-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
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
  padding: 20px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
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