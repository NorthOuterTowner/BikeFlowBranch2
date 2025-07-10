<script setup>
import { ref, onMounted, nextTick } from 'vue'
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
import Fill from 'ol/style/Fill'
import Icon from 'ol/style/Icon'
import Text from 'ol/style/Text'
import StationInfo from '../../views/dashboard/stationInfo.vue'

const mapContainer = ref(null)
const mapInstance = ref(null)
let vectorLayer = null
const router = useRouter()

// 状态管理
const stations = ref([])
const stationStatusMap = ref({})  // key: station_id, value: { stock, inflow, outflow }
const loading = ref(false)
const welcoming = ref('管理员，欢迎您！')
const searchQuery = ref('')
const fixedDate = '2025-06-12'
const currentHour = new Date().getHours()
const selectedHour = ref(currentHour.toString().padStart(2, '0'))
const showStationInfoDialog = ref(false)
const selectedStation = ref(null)

function getStationStyle(bikeNum = 0) {
  let iconSrc = '/icons/BlueLocationRound.svg'
  if (bikeNum > 10) {
    iconSrc = '/icons/RedLocationRound.svg'
  } else if (bikeNum > 9) {
    iconSrc = '/icons/YellowLocationRound.svg'
  }

  return new Style({
    image: new Icon({
      src: iconSrc,
      scale: 1.5,
      anchor: [0.5, 1]
    }),
    text: new Text({
      text: bikeNum.toString(),
      fill: new Fill({ color: '#ffffff' }),
      font: '12px Arial',
      offsetY: -20
    })
  })
}

/**
 * 获取所有站点位置
 */
async function fetchStationLocations() {
  console.log('进到获取站点位置函数')
  try {
    loading.value = true
    const response = await request.get('/stations/locations')
    
    // 处理可能的不同响应结构
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

/**
 * 获取指定时间所有站点的单车数量和流量
 * @param predictTime 预测时间，格式为 'HH:mm:ss'(疑似)
 */
 async function fetchAllStationsStatus(predictTime) {
  try {
    loading.value = true
    const res = await request.get('/predict/stations/all', {
      params: { predict_time: predictTime }
    })
    // 转成 { station_id: { stock, inflow, outflow } }
    const newMap = {}
    res.data.stations_status.forEach(item => {
      newMap[item.station_id] = {
        stock: item.stock,
        inflow: item.inflow,
        outflow: item.outflow
      }
    })
    stationStatusMap.value = newMap
    console.log('站点状态:', stationStatusMap.value)
    updateMapDisplay()
      } catch (error) {
    console.error('获取所有站点状态失败:', error.response || error.message ||error)
    console.log('请求地址:', '/predict/stations/all', '参数:', predictTime)

  } finally {
    loading.value = false
  }
}

// async function fetchStationBikeNum(stationId, date, hour) {
//   try {
//     const response = await request.get('/stations/bikeNum', {
//       params: { 
//         station_id: stationId, 
//         date: date, 
//         hour: hour 
//       }
//     })
//     // 处理不同的响应格式
//     if (response.data && typeof response.data.bikeNum === 'number') {
//       return response.data.bikeNum
//     } else if (typeof response.data === 'number') {
//       return response.data
//     } else {
//       console.warn(`站点 ${stationId} 返回数据格式异常:`, response.data)
//       return 0
//     }
//   } catch (error) {
//     console.error(`获取站点 ${stationId} 单车数量失败:`, error)
//     return 0
//   }
// }

// async function fetchAllStationsBikeNum(date, hour) {
//   if (!stations.value || stations.value.length === 0) {
//     console.warn('没有站点数据，跳过获取单车数量')
//     return
//   }

//   loading.value = true
//   console.log(`开始获取 ${date} ${hour}:00 的单车数量数据`)
  
//   try {
//     const bikeCounts = new Map()
//     const promises = stations.value.map(async station => {
//       const bikeNum = await fetchStationBikeNum(station.station_id, date, hour)
//       bikeCounts.set(station.station_id, bikeNum)
//       return { stationId: station.station_id, bikeNum }
//     })
//     const results = await Promise.all(promises)
//     console.log('单车数量获取结果:', results)
//     stationBikeCounts.value = bikeCounts
//     // 更新地图显示
//     updateMapDisplay()
//   } catch (error) {
//     console.error('获取站点单车数量失败:', error)
//   } finally {
//     loading.value = false
//   }
// }


// 地图相关函数
function initializeMap() {
  if (!mapContainer.value) {
    console.error('地图容器未找到')
    return
  }

  mapInstance.value = new Map({
    target: mapContainer.value,
    layers: [
      new TileLayer({
        source: new OSM()
      })
    ],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]), // 纽约坐标
      zoom: 11,
      maxZoom: 20,
      minZoom: 3
    })
  })

  // 创建矢量图层
  vectorLayer = new VectorLayer({
    source: new VectorSource()
  })
  mapInstance.value.addLayer(vectorLayer)

  // 绑定点击事件
  mapInstance.value.on('singleclick', onMapClick)
  
  console.log('地图初始化完成')
}

function updateMapDisplay() {
  if (!mapInstance || !vectorLayer || !stations.value.length) {
    console.warn('地图未初始化或没有站点数据')
    return
  }

  // 清除现有要素
  vectorLayer.getSource().clear()

  // 创建新的要素
  const features = stations.value.map(station => {
    // 验证坐标数据
    if (!station.longitude || !station.latitude) {
      console.warn('站点坐标数据缺失:', station)
      return null
    }
    const status = stationStatusMap.value[station.station_id] || {}
    const bikeNum = status.stock ?? 0
    const feature = new Feature({
      geometry: new Point(fromLonLat([
        parseFloat(station.longitude), 
        parseFloat(station.latitude)
      ]))
    })
    feature.setStyle(getStationStyle(bikeNum))
    feature.set('stationData', { ...station, bikeNum })
    return feature
  }).filter(Boolean) // 过滤掉空值
  // 添加要素到图层
  vectorLayer.getSource().addFeatures(features)
  console.log(`已添加 ${features.length} 个站点到地图`)
  console.log('当前 vectorLayer 中要素数量:', vectorLayer.getSource().getFeatures().length)

}

// 事件处理函数
function onMapClick(evt) {
if (!mapInstance) return
mapInstance.value.forEachFeatureAtPixel(evt.pixel, function(feature) {
  const station = feature.get('stationData')
  if (station) {
    console.log('点击了站点:', station)
    selectedStation.value = station
    showStationInfoDialog.value = true
    return true // 停止遍历
  }
})
}

const handleDialogClose = () => {
showStationInfoDialog.value = false
selectedStation.value = null
}

const handleHourChange = async () => {
  const predictTime = `${fixedDate}T${selectedHour.value}:00:00Z`
  console.log('时间变更为:', selectedHour.value)
  await fetchAllStationsStatus(predictTime)
}

function handleSearch() {
  const query = searchQuery.value.trim()
  if (!query) return
  const matched = stations.find(s =>
    s.station_name.toLowerCase().includes(query.toLowerCase()) ||
    s.station_id.toLowerCase().includes(query.toLowerCase())
  )
  if (matched) {
    mapInstance.value.getView().animate({
      center: fromLonLat([parseFloat(matched.longitude), parseFloat(matched.latitude)]),
      zoom: 15,
      duration: 1000
    })
  } else {
    alert('未找到相关站点')
  }
}


/**
 * 登出功能
 */
const logout = async () => {
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，继续执行跳转', error)
  } finally {
    router.push('/login')
  }
}

/**
 * 初始化
 * 组件挂载时获取站点位置和初始化地图
 */
onMounted(async () => {
  try {
    // 等待 DOM 渲染完成
    await nextTick()
    // 初始化地图
    initializeMap()
    // 获取站点数据
    await fetchStationLocations()
    const predictTime = `${fixedDate}T${selectedHour.value}:00:00Z`
    await fetchAllStationsStatus(predictTime)
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
  align-items: flex-start;
  margin-left: 20px;
  gap: 20px;
  flex-shrink: 0; 
}

.user-top {
  display: flex;
  align-items: center;
  gap: 150px;
}

.welcoming {
  font-size: 14px;
  white-space: nowrap; 
}

.logout-button, .search-button {
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.2s;
}

.logout-button:hover, .search-button:hover {
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