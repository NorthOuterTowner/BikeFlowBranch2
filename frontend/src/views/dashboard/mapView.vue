<script setup>
import { ref, onMounted } from 'vue'
import request from '../../api/axios'
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
const stationBikeCounts = ref(new Map())
const loading = ref(false)

function getColorByBikeCount(count) {
  if (count >= 50) return '#1a5490'
  else if (count >= 20) return '#4a90e2'
  else return '#87ceeb'
}

function getStationStyle(station) {
  const bikeCount = stationBikeCounts.value.get(station.station_id) || 0
  const color = getColorByBikeCount(bikeCount)

  return new Style({
    image: new Circle({
      radius: 8,
      fill: new Fill({ color }),
      stroke: new Stroke({ color: '#ffffff', width: 2 })
    }),
    text: new Text({
      text: bikeCount.toString(),
      fill: new Fill({ color: '#ffffff' }),
      font: '12px Arial',
      textAlign: 'center',
      textBaseline: 'middle'
    })
  })
}

async function fetchStationLocations() {
  try {
    const response = await request.get('/stations/locations')
    stations.value = response.data || response.data.data
    console.log('站点位置数据:', stations.value)
  } catch (error) {
    console.error('获取站点位置失败:', error)
  }
}

async function fetchStationBikeNum(stationId, date, hour) {
  try {
    const response = await request.get('/stations/bikeNum', {
      params: { station_id: stationId, date, hour }
    })
    return response.data.bikeNum || 0
  } catch (error) {
    console.error(`获取站点 ${stationId} 单车数量失败:`, error)
    return 0
  }
}

async function fetchAllStationsBikeNum(date, hour) {
  if (!stations.value.length) return

  loading.value = true
  const bikeCounts = new Map()

  try {
    const promises = stations.value.map(async station => {
      const bikeNum = await fetchStationBikeNum(station.station_id, date, hour)
      bikeCounts.set(station.station_id, bikeNum)
    })
    await Promise.all(promises)
    stationBikeCounts.value = bikeCounts
    updateMapDisplay()
  } catch (error) {
    console.error('获取站点单车数量失败:', error)
  } finally {
    loading.value = false
  }
}

function updateMapDisplay() {
  if (!mapInstance || !vectorLayer) return
  vectorLayer.getSource().clear()

  const features = stations.value.map(station => {
    const feature = new Feature({
      geometry: new Point(fromLonLat([station.longitude, station.latitude]))
    })
    feature.setStyle(getStationStyle(station))
    feature.set('stationData', station)
    return feature
  })

  vectorLayer.getSource().addFeatures(features)
}

const fixedDate = '2025-01-25'
const currentHour = new Date().getHours()
const selectedHour = ref(currentHour.toString().padStart(2, '0'))
const handleHourChange = async () => {
  await fetchAllStationsBikeNum(fixedDate, selectedHour.value)
}

const welcoming = ref('管理员，欢迎您！')
const searchQuery = ref('')

const handleSearch = async () => {
  if (!searchQuery.value.trim()) return

  // 调用 Nominatim API 进行地名搜索
  const url = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(searchQuery.value)}`
  try {
    const res = await fetch(url)
    const results = await res.json()
    if (results && results.length > 0) {
      const { lat, lon } = results[0]
      mapInstance.getView().animate({
        center: fromLonLat([parseFloat(lon), parseFloat(lat)]),
        zoom: 15,
        duration: 1000
      })
    } else {
      alert('未找到相关地点')
    }
  } catch (e) {
    console.error('搜索地点失败:', e)
  }
}

const logout = async () => {
  try {
    await axios.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  }
  router.push('/login')
}

onMounted(async () => {
  await fetchStationLocations()

  mapInstance = new Map({
    target: mapContainer.value,
    layers: [new TileLayer({ source: new OSM() })],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]),
      zoom: 11,
      maxZoom: 20,
      minZoom: 3
    })
  })

  vectorLayer = new VectorLayer({ source: new VectorSource() })
  mapInstance.addLayer(vectorLayer)

  stations.value.forEach(station => {
    stationBikeCounts.value.set(station.station_id, 0)
  })
  updateMapDisplay()
  await fetchAllStationsBikeNum(fixedDate, selectedHour.value)
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
.date-input,
.hour-select {
  padding: 4px 8px;
  font-size: 14px;
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

.right-time .fixed-date {
  margin-right: 40px;
}

.right-time select {
  padding: 6px 10px;
  font-size: 16px;
  height: 30px;
  border-radius: 4px;
}

.disabled-option {
  color: #999;
  background-color: #f2f2f2;
}


</style>