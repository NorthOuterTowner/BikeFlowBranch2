<script setup>
import { ref, onMounted, nextTick, computed } from 'vue'
import request from '../../api/axios'
import { useRouter } from 'vue-router'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import { fromLonLat } from 'ol/proj'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import LineString from 'ol/geom/LineString'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Style from 'ol/style/Style'
import Fill from 'ol/style/Fill'
import Stroke from 'ol/style/Stroke'
import Icon from 'ol/style/Icon'
import Text from 'ol/style/Text'
import {Zoom } from 'ol/control'

const mapContainer = ref(null)
let mapInstance = null
let vectorLayer = null
let dispatchLayer = null //调度方案图层
const router = useRouter()

// 状态管理
const stations = ref([])
const stationStatusMap = ref({})  // key: station_id, value: { stock, inflow, outflow }
const loading = ref(false)
const welcoming = ref('管理员，欢迎您！')
const searchQuery = ref('')

// 调度方案相关状态
const showDispatchLayer = ref(false) // 是否显示调度图层
const dispatchPlans = ref([]) // 调度方案数据

// 悬停提示相关
const tooltip = ref(null)
const showTooltip = ref(false)
const tooltipContent = ref('')
const tooltipPosition = ref({ x: 0, y: 0 })
const fixedDate = computed(() => {
  return localStorage.getItem('selectedDate') || new Date().toISOString().split('T')[0]
})
const currentHour = getCurrentHourString()

function getCurrentHourString() {
  const now = new Date()
  const hour = now.getHours().toString().padStart(2, '0')
  return `${hour}:00`
}

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
 * 创建调度箭头样式
 * @param {number} quantity - 调度数量
 * @param {string} color - 箭头颜色
 * @returns {Style} OpenLayers样式对象
 */
function createDispatchArrowStyle(quantity, color = '#ff6b35') {
  // 根据调度数量计算线条宽度 (最小2px，最大10px)
  const lineWidth = Math.max(2, Math.min(10, quantity * 0.8))
  
  return new Style({
    stroke: new Stroke({
      color: color,
      width: lineWidth,
      lineDash: [0] // 实线
    }),
    text: new Text({
      text: `${quantity}`,
      fill: new Fill({ color: '#ffffff' }),
      stroke: new Stroke({ color: color, width: 2 }),
      font: 'bold 12px Arial',
      placement: 'line',
      textAlign: 'center',
      offsetY: -2
    })
  })
}

/**
 * 创建箭头头部样式
 * @param {Array} endCoordinate - 终点坐标
 * @param {number} rotation - 旋转角度
 * @param {string} color - 箭头颜色
 * @returns {Style} 箭头头部样式
 */
function createArrowHeadStyle(endCoordinate, rotation, color = '#ff6b35') {
  return new Style({
    geometry: new Point(endCoordinate),
    image: new Icon({
      src: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
          <path d="M10 2 L18 10 L10 18 L12 10 Z" fill="${color}" stroke="white" stroke-width="1"/>
        </svg>
      `),
      scale: 0.8,
      rotation: rotation,
      anchor: [0.5, 0.5]
    })
  })
}

/**
 * 计算两点之间的角度
 * @param {Array} start - 起点坐标
 * @param {Array} end - 终点坐标
 * @returns {number} 角度（弧度）
 */
function calculateAngle(start, end) {
  const dx = end[0] - start[0]
  const dy = end[1] - start[1]
  return Math.atan2(dy, dx)
}

/**
 * 添加调度方案到地图
 * @param {Array} dispatches - 调度方案数组
 * 每个元素格式: { startStationId, endStationId, quantity }
 */
function addDispatchesToMap(dispatches) {
  if (!mapInstance || !dispatchLayer || !stations.value.length) {
    console.warn('地图未初始化或缺少必要数据')
    return
  }

  // 清除现有的调度箭头
  dispatchLayer.getSource().clear()

  const features = []

  dispatches.forEach(dispatch => {
    const { startStationId, endStationId, quantity } = dispatch

    // 查找起点和终点站点
    const startStation = stations.value.find(s => s.station_id === startStationId)
    const endStation = stations.value.find(s => s.station_id === endStationId)

    if (!startStation || !endStation) {
      console.warn(`找不到站点: ${startStationId} 或 ${endStationId}`)
      return
    }

    // 转换坐标
    const startCoord = fromLonLat([parseFloat(startStation.longitude), parseFloat(startStation.latitude)])
    const endCoord = fromLonLat([parseFloat(endStation.longitude), parseFloat(endStation.latitude)])

    // 创建线条要素
    const lineFeature = new Feature({
      geometry: new LineString([startCoord, endCoord])
    })

    // 设置线条样式
    const lineStyle = createDispatchArrowStyle(quantity)
    lineFeature.setStyle(lineStyle)

    // 设置要素属性（用于悬停提示）
    lineFeature.set('dispatchData', {
      startStation: startStation.station_name || startStationId,
      endStation: endStation.station_name || endStationId,
      quantity: quantity
    })

    features.push(lineFeature)

    // 创建箭头头部
    const angle = calculateAngle(startCoord, endCoord)
    const arrowHeadFeature = new Feature({
      geometry: new Point(endCoord)
    })
    
    const arrowHeadStyle = createArrowHeadStyle(endCoord, angle)
    arrowHeadFeature.setStyle(arrowHeadStyle)
    
    features.push(arrowHeadFeature)
  })

  // 添加要素到图层
  dispatchLayer.getSource().addFeatures(features)
  console.log(`已添加 ${features.length} 个调度要素到地图`)
}

/**
 * 模拟调度方案数据（测试用）
 */
function generateMockDispatchData() {
  if (stations.value.length < 2) {
    console.warn('站点数据不足，无法生成模拟调度方案')
    return []
  }

  const mockDispatches = []
  const numDispatches = Math.min(5, Math.floor(stations.value.length / 2)) // 生成5个调度或站点数量的一半

  for (let i = 0; i < numDispatches; i++) {
    const startIndex = Math.floor(Math.random() * stations.value.length)
    let endIndex = Math.floor(Math.random() * stations.value.length)
    
    // 确保起点和终点不同
    while (endIndex === startIndex) {
      endIndex = Math.floor(Math.random() * stations.value.length)
    }

    mockDispatches.push({
      startStationId: stations.value[startIndex].station_id,
      endStationId: stations.value[endIndex].station_id,
      quantity: Math.floor(Math.random() * 15) + 1 // 1-15的随机数量
    })
  }

  return mockDispatches
}

/**
 * 切换调度图层显示状态
 */
function toggleDispatchLayer() {
  showDispatchLayer.value = !showDispatchLayer.value
  
  if (showDispatchLayer.value) {
    // 显示调度图层
    if (dispatchPlans.value.length === 0) {
      // 如果没有调度方案数据，生成模拟数据
      dispatchPlans.value = generateMockDispatchData()
    }
    addDispatchesToMap(dispatchPlans.value)
    dispatchLayer.setVisible(true)
  } else {
    // 隐藏调度图层
    dispatchLayer.setVisible(false)
  }
}

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

async function fetchAllStationsStatus(date, hour) {
  const startTime = Date.now()
  
  try {
    loading.value = true
    stationStatusMap.value = {}
    
    console.log('=== 请求开始 ===')
    console.log('开始时间:', new Date().toISOString())
    console.log('请求参数:', { date, hour })
    
    // 检查网络状态
    if (!navigator.onLine) {
      console.error('网络离线状态')
      return
    }
    
    // 构建完整的请求URL用于调试
    const baseURL = request.defaults?.baseURL || ''
    const fullURL = `${baseURL}/stations/bikeNum/timeAll?date=${date}&hour=${hour}`
    console.log('完整请求URL:', fullURL)
    
    // 发送请求前的时间戳
    const requestStartTime = Date.now()
    console.log('发送请求时间戳:', requestStartTime)
    
    const res = await request.get('/stations/bikeNum/timeAll', {
      params: {
        date: date,
        hour: hour
      },
      timeout: 30000,
      // 添加请求拦截器来确认请求是否发送
      onUploadProgress: (progressEvent) => {
        console.log('请求上传进度:', progressEvent)
      },
      onDownloadProgress: (progressEvent) => {
        console.log('响应下载进度:', progressEvent)
      }
    })
    
    const requestEndTime = Date.now()
    const requestDuration = requestEndTime - requestStartTime
    console.log('请求完成时间戳:', requestEndTime)
    console.log('请求耗时:', requestDuration + 'ms')
    
    console.log('API响应状态:', res.status)
    console.log('API响应头:', res.headers)
    console.log('API响应数据:', res.data)
    
    if (res.data && res.data.code === 200 && res.data.rows && Array.isArray(res.data.rows)) {
      const newMap = {}
      res.data.rows.forEach(item => {
        newMap[item.station_id] = {
          stock: item.stock || 0,
          inflow: 0,
          outflow: 0
        }
      })
      stationStatusMap.value = newMap
      console.log('站点状态:', stationStatusMap.value)
      console.log(`成功获取到 ${res.data.rows.length} 个站点的单车数量数据`)
    } else {
      console.warn('没有获取到有效的站点状态数据')
      stationStatusMap.value = {}
    }
    
    updateMapDisplay()
    
  } catch (error) {
    const errorTime = Date.now()
    const totalDuration = errorTime - startTime
    
    console.error('=== 请求失败 ===')
    console.error('失败时间:', new Date().toISOString())
    console.error('总耗时:', totalDuration + 'ms')
    console.error('错误对象:', error)
    console.error('错误代码:', error.code)
    console.error('错误消息:', error.message)
    console.error('错误堆栈:', error.stack)
    
    // 详细分析错误类型
    if (error.code === 'ECONNABORTED') {
      console.error('❌ 请求超时 - 客户端设置的超时时间到达')
    } else if (error.message.includes('timeout')) {
      console.error('❌ 请求超时 - 网络层超时')
    } else if (error.message.includes('Network Error')) {
      console.error('❌ 网络错误 - 请求可能没有发送到服务器')
    } else if (error.response) {
      console.error('❌ 服务器响应错误')
      console.error('响应状态:', error.response.status)
      console.error('响应数据:', error.response.data)
    } else if (error.request) {
      console.error('❌ 请求已发送但没有收到响应')
      console.error('请求对象:', error.request)
    } else {
      console.error('❌ 未知错误')
    }
    
    stationStatusMap.value = {}
    updateMapDisplay()
    
  } finally {
    loading.value = false
    const endTime = Date.now()
    const totalTime = endTime - startTime
    console.log('=== 请求结束 ===')
    console.log('结束时间:', new Date().toISOString())
    console.log('总执行时间:', totalTime + 'ms')
  }
}


function initializeMap() {
  if (!mapContainer.value) {
    console.error('地图容器未找到')
    return
  }

  mapInstance = new Map({
    target: mapContainer.value,
    layers: [
      new TileLayer({
        source: new OSM()
      })
    ],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]),
      zoom: 11,
      maxZoom: 20,
      minZoom: 3
    }),
    controls: [] 

  })

  // 站点图层
  vectorLayer = new VectorLayer({
    source: new VectorSource()
  })

  // 调度方案图层
  dispatchLayer = new VectorLayer({
    source: new VectorSource(),
    visible: false // 默认隐藏
  })

  mapInstance.addLayer(vectorLayer)
  mapInstance.addLayer(dispatchLayer)

  // 添加鼠标移动事件监听器用于悬停提示
  mapInstance.on('pointermove', onMapHover)

  console.log('地图初始化完成')
}

/**
 * 更新地图显示
 */
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
}

// 鼠标悬停事件处理函数
function onMapHover(evt) {
  if (!mapInstance) return
  
  const pixel = mapInstance.getEventPixel(evt.originalEvent)
  const feature = mapInstance.forEachFeatureAtPixel(pixel, function(feature) {
    return feature
  })

  if (feature) {
    const station = feature.get('stationData')
    const dispatchData = feature.get('dispatchData')
    
    if (station) {
      // 显示站点悬停提示
      const status = stationStatusMap.value[station.station_id] || {}
      const bikeNum = status.stock ?? 0
      tooltipContent.value = `${station.station_name || station.station_id}`
      tooltipPosition.value = {
        x: evt.originalEvent.clientX + 10,
        y: evt.originalEvent.clientY - 10
      }
      showTooltip.value = true
      mapInstance.getTargetElement().style.cursor = 'pointer'
    } else if (dispatchData) {
      // 显示调度信息悬停提示
      tooltipContent.value = `${dispatchData.startStation} → ${dispatchData.endStation} (${dispatchData.quantity}辆)`
      tooltipPosition.value = {
        x: evt.originalEvent.clientX + 10,
        y: evt.originalEvent.clientY - 10
      }
      showTooltip.value = true
      mapInstance.getTargetElement().style.cursor = 'pointer'
    }
  } else {
    // 隐藏悬停提示
    showTooltip.value = false
    mapInstance.getTargetElement().style.cursor = ''
  }
}

/**
 * 搜索站点功能
 */
 const handleSearch = () => {
  console.log('搜索按钮被点击，搜索词:', searchQuery.value)
  
  if (!searchQuery.value.trim()) {
    console.log('搜索词为空')
    alert('请输入搜索内容')
    return
  }
  
  if (!stations.value || stations.value.length === 0) {
    console.log('没有站点数据')
    alert('站点数据未加载')
    return
  }
  
  if (!mapInstance) {
    console.log('地图实例未初始化')
    alert('地图未初始化')
    return
  }
  
  console.log('开始搜索，当前站点数量:', stations.value.length)
  
  const matchedStations = stations.value.filter(station => {
    const stationName = station.station_name || ''
    const stationId = station.station_id || ''
    const searchTerm = searchQuery.value.toLowerCase().trim()
    
    return stationName.toLowerCase().includes(searchTerm) ||
           stationId.toLowerCase().includes(searchTerm)
  })
  
  console.log('匹配到的站点:', matchedStations)
  
  if (matchedStations.length > 0) {
    const station = matchedStations[0]
    console.log('选中的站点:', station)
    
    // 检查坐标是否有效
    const longitude = parseFloat(station.longitude)
    const latitude = parseFloat(station.latitude)
    
    if (isNaN(longitude) || isNaN(latitude)) {
      console.error('站点坐标无效:', station)
      alert('站点坐标数据有误')
      return
    }
    
    try {
      mapInstance.getView().animate({
        center: fromLonLat([longitude, latitude]),
        zoom: 20,
        duration: 1000
      })
      console.log('地图动画执行成功')
    } catch (error) {
      console.error('地图动画执行失败:', error)
      alert('地图导航失败')
    }
  } else {
    console.log('未找到匹配的站点')
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
    console.warn('登出失败，可忽略', error)
  } finally {
    router.push('/login')
  }
}

function getCurrentHourString2() {
  const now = new Date()
  return now.getHours().toString() // 返回 "9" 而不是 "09:00"
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
    
  const zoomControl = new Zoom({
  className: 'ol-zoom-custom'
})
    mapInstance.addControl(zoomControl)
    // 获取站点数据
    await fetchStationLocations()
    await fetchAllStationsStatus(fixedDate.value,getCurrentHourString2())
  } catch (error) {
    console.error('组件初始化失败:', error)
  }
})

// 暴露方法供外部调用
defineExpose({
  addDispatchesToMap,
  toggleDispatchLayer,
  generateMockDispatchData
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
          <span class="fixed-time">{{ currentHour }}</span>
        </div>
      </div>
    </header>

    <!-- 控制面板 -->
    <div class="control-panel">
      <button 
        class="dispatch-toggle-btn" 
        :class="{ active: showDispatchLayer }"
        @click="toggleDispatchLayer"
      >
        {{ showDispatchLayer ? '隐藏调度方案' : '显示调度方案' }}
      </button>
      <span class="dispatch-info" v-if="showDispatchLayer">
        当前显示 {{ dispatchPlans.length }} 条调度路线
      </span>
    </div>

    <!-- 图例 -->
    <div class="legend">
      <div class="legend-section">
        <h4>站点状态</h4>
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
      
      <div class="legend-section" v-if="showDispatchLayer">
        <h4>调度方案</h4>
        <div class="legend-item">
          <div class="dispatch-line thin"></div>
          <span>少量调度（1-5辆）</span>
        </div>
        <div class="legend-item">
          <div class="dispatch-line medium"></div>
          <span>中量调度（6-10辆）</span>
        </div>
        <div class="legend-item">
          <div class="dispatch-line thick"></div>
          <span>大量调度（11+辆）</span>
        </div>
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
    
    <!-- 悬停提示框 -->
    <div 
      v-if="showTooltip" 
      class="tooltip"
      :style="{ 
        left: tooltipPosition.x + 'px', 
        top: tooltipPosition.y + 'px' 
      }"
    >
      {{ tooltipContent }}
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

.right-time .fixed-time {
  font-weight: bold;
  color: #091275;
}

/* 控制面板样式 */
.control-panel {
  position: absolute;
  top: 120px;
  left: 20px;
  background-color: rgba(255, 255, 255, 0.95);
  padding: 12px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 12px;
}

.dispatch-toggle-btn {
  padding: 8px 16px;
  background-color: #ff6b35;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.2s;
}

.dispatch-toggle-btn:hover {
  background-color: #e55a2b;
}

.dispatch-toggle-btn.active {
  background-color: #28a745;
}

.dispatch-toggle-btn.active:hover {
  background-color: #218838;
}

.dispatch-info {
  font-size: 12px;
  color: #666;
  font-style: italic;
}

.legend {
  position: absolute;
  top: 120px;
  right: 20px;
  background-color: rgba(255, 255, 255, 0.95);
  padding: 12px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 10;
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 200px;
}

.legend-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.legend-section h4 {
  margin: 0;
  font-size: 14px;
  color: #333;
  border-bottom: 1px solid #eee;
  padding-bottom: 4px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

.dispatch-line {
  width: 24px;
  height: 3px;
  background-color: #ff6b35;
  border-radius: 2px;
}

.dispatch-line.thin {
  height: 2px;
}

.dispatch-line.medium {
  height: 4px;
}

.dispatch-line.thick {
  height: 6px;
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

/* 悬停提示框样式 */
.tooltip {
  position: fixed;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 12px;
  white-space: nowrap;
  pointer-events: none;
  z-index: 1000;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}
</style>
