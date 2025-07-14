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
import { Zoom } from 'ol/control'

const mapContainer = ref(null)
let mapInstance = null
let vectorLayer = null
let navigationLayer = null // 导航路线图层
let routeLayer = null // 路线图层
const router = useRouter()

// 状态管理
const stations = ref([])
const stationStatusMap = ref({})
const loading = ref(false)
const welcoming = ref('管理员，欢迎您！')
const searchQuery = ref('')

// 调度方案相关状态
const dispatchPlans = ref([])
const dispatchLoading = ref(false)
const dispatchError = ref(null)

// 导航相关状态
const navigationActive = ref(false)
const currentRoute = ref(null)
const navigationInstructions = ref([])
const routeDistance = ref(0)
const routeDuration = ref(0)
const selectedDispatch = ref(null)

// 悬停提示相关
const tooltip = ref(null)
const showTooltip = ref(false)
const tooltipContent = ref('')
const tooltipPosition = ref({ x: 0, y: 0 })

const fixedDate = computed(() => {
  return localStorage.getItem('selectedDate') || new Date().toISOString().split('T')[0]
})
const currentHour = "09:00"

// OpenRouteService 配置
const ORS_API_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImE0ZjM4NDNiZmE3NDQ0YTM4MmNhNmEyMWM4NWUxYjU0IiwiaCI6Im11cm11cjY0In0=' 
const ORS_BASE_URL = 'https://api.openrouteservice.org/v2'

/**
 * 获取调度方案数据
 */
async function fetchDispatchPlans(queryTime) { 
  try {     
    dispatchLoading.value = true     
    dispatchError.value = null          
    
    console.log('获取调度方案数据，查询时间:', queryTime)          
    
    if (!queryTime || typeof queryTime !== 'string') {       
      throw new Error('无效的查询时间格式')     
    }          
    
    const response = await request.get('/dispatch', {       
      params: {         
        query_time: queryTime       
      },       
      timeout: 10000     
    })          
    
    console.log('调度方案API响应:', response.data)          
    
    if (!response.data) {       
      throw new Error('API响应数据为空')     
    }          
    
    if (!response.data.schedules || !Array.isArray(response.data.schedules)) {       
      console.warn('没有获取到有效的调度方案数据')       
      dispatchPlans.value = []       
      return []     
    }          
    
    // 添加详细调试信息 - 查看第一个调度方案的数据结构     
    if (response.data.schedules.length > 0) {       
      console.log('第一个调度方案的数据结构:', response.data.schedules[0])       
      console.log('所有字段名:', Object.keys(response.data.schedules[0]))     
    }          
    
    // 修复验证逻辑 - 使用正确的字段名     
    const validSchedules = response.data.schedules.filter(schedule => {       
      // 记录每个字段的验证结果       
      const hasStartStation = schedule.start_station !== undefined && schedule.start_station !== null       
      const hasEndStation = schedule.end_station !== undefined && schedule.end_station !== null       
      const hasBikesToMove = schedule.bikes_to_move !== undefined && schedule.bikes_to_move !== null  // 修复：使用正确的字段名
      const hasScheduleId = schedule.schedule_id !== undefined && schedule.schedule_id !== null              
      
      console.log('验证调度方案:', {         
        schedule_id: schedule.schedule_id,         
        hasStartStation,         
        hasEndStation,         
        hasBikesToMove,         
        hasScheduleId,         
        start_station: schedule.start_station,         
        end_station: schedule.end_station,         
        bikes_to_move: schedule.bikes_to_move 
      })              
      
      return hasStartStation && hasEndStation && hasBikesToMove && hasScheduleId     
    })          
    
    if (validSchedules.length !== response.data.schedules.length) {       
      console.warn(`过滤掉了 ${response.data.schedules.length - validSchedules.length} 个无效的调度方案`)     
    }          
    
    // 转换调度数据格式     
    const convertedDispatches = validSchedules.map(schedule => ({       
      startStationId: schedule.start_station.id || schedule.start_station,       
      endStationId: schedule.end_station.id || schedule.end_station,       
      quantity: schedule.bikes_to_move,  // 修复：使用正确的字段名
      scheduleId: schedule.schedule_id,       
      status: schedule.status || '待执行',       
      startStationName: schedule.start_station.name || schedule.start_station.id || schedule.start_station,       
      endStationName: schedule.end_station.name || schedule.end_station.id || schedule.end_station,       
      updatedAt: schedule.updated_at,       
      // 添加坐标信息（如果存在）       
      startStationLat: schedule.start_station?.lat,       
      startStationLng: schedule.start_station?.lng,       
      endStationLat: schedule.end_station?.lat,       
      endStationLng: schedule.end_station?.lng     
    }))          
    
    dispatchPlans.value = convertedDispatches     
    console.log(`成功获取到 ${convertedDispatches.length} 条调度方案`)     
    console.log('转换后的调度数据:', convertedDispatches)          
    
    return convertedDispatches        
    
  } catch (error) {     
    console.error('获取调度方案失败:', error)     
    // ... 错误处理代码保持不变   
  } finally {     
    dispatchLoading.value = false   
  } 
}

/**
 * 构建查询时间字符串
 */
function buildQueryTime(date, hour) {
  try {
    let hourStr = '';
    // 如果 hour 已经是形如 "HH:mm" 的字符串，直接用
    if (typeof hour === 'string' && /^\d{1,2}:\d{2}$/.test(hour)) {
      hourStr = hour.padStart(5, '0'); // 确保格式是 05:00
    } else {
      // 否则当做数字处理，只取小时部分，格式化为两位数 + ":00"
      const hourNum = parseInt(hour, 10);
      if (isNaN(hourNum) || hourNum < 0 || hourNum > 23) {
        throw new Error('小时参数无效');
      }
      hourStr = hourNum.toString().padStart(2, '0') + ':00';
    }
    return `${date}T${hourStr}:00Z`;
  } catch (error) {
    console.error('构建查询时间失败:', error);
    return new Date().toISOString();
  }
}



/**
 * 调用 OpenRouteService 获取路线
 */
async function getRoute(startCoord, endCoord) {
  try {
    const response = await fetch(`${ORS_BASE_URL}/directions/driving-car/geojson`, {
      method: 'POST',
      headers: {
        'Accept': 'application/json, application/geo+json',
        'Authorization': ORS_API_KEY,
        'Content-Type': 'application/json; charset=utf-8'
      },
      body: JSON.stringify({
        coordinates: [startCoord, endCoord],
        format: 'geojson',
        instructions: true,
        language: 'zh-cn'
      })
    })

    const raw = await response.text()
    console.log('ORS 原始响应:', raw)

    const data = JSON.parse(raw)
    return data
  } catch (error) {
    console.error('获取路线失败:', error)
    throw error
  }
}


/**
 * 显示导航路线
 */
async function showNavigation(dispatch) {
  try {
    loading.value = true
    selectedDispatch.value = dispatch
    
    // 获取起点和终点坐标
    let startCoord, endCoord
    console.log('startCoord:', startCoord)
    console.log('endCoord:', endCoord)

    
    if (dispatch.startStationLat && dispatch.startStationLng && dispatch.endStationLat && dispatch.endStationLng) {
      startCoord = [parseFloat(dispatch.startStationLng), parseFloat(dispatch.startStationLat)]
      endCoord = [parseFloat(dispatch.endStationLng), parseFloat(dispatch.endStationLat)]
    } else {
      const startStation = stations.value.find(s => s.station_id === dispatch.startStationId)
      const endStation = stations.value.find(s => s.station_id === dispatch.endStationId)
      
      if (!startStation || !endStation) {
        throw new Error('找不到站点坐标信息')
      }
      
      startCoord = [parseFloat(startStation.longitude), parseFloat(startStation.latitude)]
      endCoord = [parseFloat(endStation.longitude), parseFloat(endStation.latitude)]
    }
    
    console.log('开始导航:', { startCoord, endCoord })
    
    // 获取路线数据
    const routeData = await getRoute(startCoord, endCoord)
    
    if (!routeData.features || routeData.features.length === 0) {
      throw new Error('未找到有效路线')
    }
    
    const route = routeData.features[0]
    const geometry = route.geometry
    const properties = route.properties
    
    // 保存路线信息
    currentRoute.value = route
    navigationInstructions.value = properties.segments[0].steps || []
    routeDistance.value = (properties.summary.distance / 1000).toFixed(2) // 转换为公里
    routeDuration.value = Math.round(properties.summary.duration / 60) // 转换为分钟
    
    // 清除现有导航图层
    navigationLayer.getSource().clear()
    
    // 创建路线要素
    const routeCoords = geometry.coordinates.map(coord => fromLonLat(coord))
    const routeFeature = new Feature({
      geometry: new LineString(routeCoords)
    })
    
    // 设置路线样式
    routeFeature.setStyle(new Style({
      stroke: new Stroke({
        color: '#007bff',
        width: 4
      })
    }))
    
    // 添加起点标记
    const startMarker = new Feature({
      geometry: new Point(fromLonLat(startCoord))
    })
    startMarker.setStyle(new Style({
      image: new Icon({
        src: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="#28a745" stroke="white" stroke-width="2"/>
            <text x="12" y="17" text-anchor="middle" fill="white" font-size="12" font-weight="bold">起</text>
          </svg>
        `),
        scale: 1.2,
        anchor: [0.5, 1]
      })
    }))
    
    // 添加终点标记
    const endMarker = new Feature({
      geometry: new Point(fromLonLat(endCoord))
    })
    endMarker.setStyle(new Style({
      image: new Icon({
        src: 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(`
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="10" fill="#dc3545" stroke="white" stroke-width="2"/>
            <text x="12" y="17" text-anchor="middle" fill="white" font-size="12" font-weight="bold">终</text>
          </svg>
        `),
        scale: 1.2,
        anchor: [0.5, 1]
      })
    }))
    
    // 添加要素到导航图层
    navigationLayer.getSource().addFeatures([routeFeature, startMarker, endMarker])
    
    // 设置地图视图以包含整个路线
    const extent = routeFeature.getGeometry().getExtent()
    mapInstance.getView().fit(extent, {
      padding: [50, 50, 50, 50],
      duration: 1000
    })
    
    navigationActive.value = true
    
    console.log('导航路线显示成功')
    
  } catch (error) {
    console.error('显示导航失败:', error)
    alert('导航失败：' + error.message)
  } finally {
    loading.value = false
  }
}

/**
 * 清除导航
 */
function clearNavigation() {
  navigationActive.value = false
  currentRoute.value = null
  navigationInstructions.value = []
  routeDistance.value = 0
  routeDuration.value = 0
  selectedDispatch.value = null
  
  if (navigationLayer) {
    navigationLayer.getSource().clear()
  }
}

/**
 * 创建调度方案列表项的点击处理
 */
function handleDispatchClick(dispatch) {
  showNavigation(dispatch)
}

/**
 * 获取站点样式
 */
function getStationStyle(bikeNum = 0) {
  let iconSrc = '/icons/BlueLocationRound.svg'
  if (bikeNum > 10) {
    iconSrc = '/icons/RedLocationRound.svg'
  } else if (bikeNum > 5) {
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
 * 获取站点位置数据
 */
async function fetchStationLocations() {
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

/**
 * 获取所有站点状态
 */
async function fetchAllStationsStatus(date, hour) {
  try {
    loading.value = true
    stationStatusMap.value = {}
    
    const res = await request.get('/stations/bikeNum/timeAll', {
      params: { date, hour },
      timeout: 30000
    })
    
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
      console.log(`成功获取到 ${res.data.rows.length} 个站点的单车数量数据`)
    } else {
      console.warn('没有获取到有效的站点状态数据')
      stationStatusMap.value = {}
    }
    
    updateMapDisplay()
    
  } catch (error) {
    console.error('获取站点状态失败:', error)
    stationStatusMap.value = {}
    updateMapDisplay()
  } finally {
    loading.value = false
  }
}

/**
 * 初始化地图
 */
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
      zoom: 14,
      maxZoom: 20,
      minZoom: 3
    }),
    controls: []
  })

  // 站点图层
  vectorLayer = new VectorLayer({
    source: new VectorSource()
  })

  // 导航图层
  navigationLayer = new VectorLayer({
    source: new VectorSource()
  })

  mapInstance.addLayer(vectorLayer)
  mapInstance.addLayer(navigationLayer)

  // 添加鼠标移动事件监听器
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

  vectorLayer.getSource().clear()

  const features = stations.value.map(station => {
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
  }).filter(Boolean)

  vectorLayer.getSource().addFeatures(features)
  console.log(`已添加 ${features.length} 个站点到地图`)
}

/**
 * 地图悬停事件处理
 */
function onMapHover(evt) {
  if (!mapInstance) return
  
  const pixel = mapInstance.getEventPixel(evt.originalEvent)
  const feature = mapInstance.forEachFeatureAtPixel(pixel, function(feature) {
    return feature
  })

  if (feature) {
    const station = feature.get('stationData')
    
    if (station) {
      const status = stationStatusMap.value[station.station_id] || {}
      const bikeNum = status.stock ?? 0
      tooltipContent.value = `${station.station_name || station.station_id} (${bikeNum}辆)`
      tooltipPosition.value = {
        x: evt.originalEvent.clientX + 10,
        y: evt.originalEvent.clientY - 10
      }
      showTooltip.value = true
      mapInstance.getTargetElement().style.cursor = 'pointer'
    }
  } else {
    showTooltip.value = false
    mapInstance.getTargetElement().style.cursor = ''
  }
}

/**
 * 搜索站点功能
 */
const handleSearch = () => {
  if (!searchQuery.value.trim()) {
    alert('请输入搜索内容')
    return
  }
  
  if (!stations.value || stations.value.length === 0) {
    alert('站点数据未加载')
    return
  }
  
  if (!mapInstance) {
    alert('地图未初始化')
    return
  }
  
  const matchedStations = stations.value.filter(station => {
    const stationName = station.station_name || ''
    const stationId = station.station_id || ''
    const searchTerm = searchQuery.value.toLowerCase().trim()
    
    return stationName.toLowerCase().includes(searchTerm) ||
           stationId.toLowerCase().includes(searchTerm)
  })
  
  if (matchedStations.length > 0) {
    const station = matchedStations[0]
    const longitude = parseFloat(station.longitude)
    const latitude = parseFloat(station.latitude)
    
    if (isNaN(longitude) || isNaN(latitude)) {
      alert('站点坐标数据有误')
      return
    }
    
    try {
      mapInstance.getView().animate({
        center: fromLonLat([longitude, latitude]),
        zoom: 18,
        duration: 1000
      })
    } catch (error) {
      console.error('地图动画执行失败:', error)
      alert('地图导航失败')
    }
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
    console.warn('登出失败，可忽略', error)
  } finally {
    router.push('/login')
  }
}

/**
 * 获取当前小时字符串
 */
function getCurrentHourString() {
  const now = new Date()
  return now.getHours().toString()
}

/**
 * 加载调度方案
 */
async function loadDispatchPlans() {
  const queryTime = buildQueryTime(fixedDate.value, '9:00')
  await fetchDispatchPlans(queryTime)
}

/**
 * 组件挂载时初始化
 */
onMounted(async () => {
  try {
    await nextTick()
    initializeMap()
    
    const zoomControl = new Zoom({
      className: 'ol-zoom-custom'
    })
    mapInstance.addControl(zoomControl)
    
    await fetchStationLocations()
    await fetchAllStationsStatus(fixedDate.value, getCurrentHourString())
    await loadDispatchPlans()
  } catch (error) {
    console.error('组件初始化失败:', error)
  }
})

defineExpose({
  showNavigation,
  clearNavigation,
  loadDispatchPlans
})
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车调度导航系统</h1>
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
          <label>当前时段：</label>
          <span class="fixed-time">{{ currentHour }}</span>
        </div>
      </div>
    </header>

    <!-- 调度方案面板 -->
    <div class="dispatch-panel">
      <div class="panel-header">
        <h3>调度方案列表</h3>
        <button class="refresh-btn" @click="loadDispatchPlans" :disabled="dispatchLoading">
          {{ dispatchLoading ? '加载中...' : '刷新' }}
        </button>
      </div>
      
      <div class="dispatch-list" v-if="dispatchPlans.length > 0">
        <div 
          v-for="dispatch in dispatchPlans" 
          :key="dispatch.scheduleId"
          class="dispatch-item"
          :class="{ active: selectedDispatch?.scheduleId === dispatch.scheduleId }"
          @click="handleDispatchClick(dispatch)"
        >
          <div class="dispatch-info">
            <div class="dispatch-id">调度 #{{ dispatch.scheduleId }}</div>
            <div class="dispatch-route">
              <span class="start-station">{{ dispatch.startStationName }}</span>
              <span class="arrow">→</span>
              <span class="end-station">{{ dispatch.endStationName }}</span>
            </div>
            <div class="dispatch-details">
              <span class="quantity">{{ dispatch.quantity }}辆</span>
              <span class="status" :class="dispatch.status">{{ dispatch.status }}</span>
            </div>
          </div>
          <div class="dispatch-action">
            <button class="nav-btn">导航</button>
          </div>
        </div>
      </div>
      
      <div v-else-if="dispatchLoading" class="loading-message">
        正在加载调度方案...
      </div>
      
      <div v-else class="empty-message">
        暂无调度方案
      </div>
      
      <div v-if="dispatchError" class="error-message">
        {{ dispatchError }}
      </div>
    </div>

    <!-- 导航信息面板 -->
    <div v-if="navigationActive" class="navigation-panel">
      <div class="nav-header">
        <h3>导航信息</h3>
        <button class="close-nav-btn" @click="clearNavigation">×</button>
      </div>
      
      <div class="nav-summary">
        <div class="nav-route">
          <span class="start">{{ selectedDispatch?.startStationName }}</span>
          <span class="arrow">→</span>
          <span class="end">{{ selectedDispatch?.endStationName }}</span>
        </div>
        <div class="nav-stats">
          <span class="distance">{{ routeDistance }}km</span>
          <span class="duration">{{ routeDuration }}分钟</span>
        </div>
      </div>
      
      <div class="nav-instructions" v-if="navigationInstructions.length > 0">
        <h4>导航指引</h4>
        <div class="instruction-list">
          <div 
            v-for="(instruction, index) in navigationInstructions.slice(0, 5)" 
            :key="index"
            class="instruction-item"
          >
            <div class="instruction-icon">{{ index + 1 }}</div>
            <div class="instruction-text">{{ instruction.instruction }}</div>
            <div class="instruction-distance">{{ (instruction.distance / 1000).toFixed(2) }}km</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 图例 -->
    <div class="legend">
      <div class="legend-section">
        <h4>站点状态</h4>
        <div class="legend-item">
          <img src="/icons/BlueLocationRound.svg" width="20" height="20" alt="少">
          <span>少（0–5）</span>
        </div>
        <div class="legend-item">
          <img src="/icons/YellowLocationRound.svg" width="20" height="20" alt="中">
          <span>中（6–10）</span>
        </div>
        <div class="legend-item">
          <img src="/icons/RedLocationRound.svg" width="20" height="20" alt="多">
          <span>多（11+）</span>
        </div>
      </div>
      
      <div class="legend-section" v-if="navigationActive">
        <h4>导航标记</h4>
        <div class="legend-item">
          <div class="nav-marker start-marker">起</div>
          <span>起点</span>
        </div>
        <div class="legend-item">
          <div class="nav-marker end-marker">终</div>
          <span>终点</span>
        </div>
        <div class="legend-item">
          <div class="route-line"></div>
          <span>导航路线</span>
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

    <!-- 地图 -->
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
.dispatch-panel {
  padding: 20px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  margin: 20px;
  flex-shrink: 0;
}
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.refresh-btn {
  padding: 6px 12px;
  background-color: #28a745;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}
.refresh-btn:hover {
  background-color: #218838;
}
.dispatch-list {
  max-height: 400px;
  overflow-y: auto;
}
.dispatch-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  border-bottom: 1px solid #ddd;
  cursor: pointer;
}
.dispatch-item:hover {
  background-color: #f8f9fa;
}
.dispatch-item.active {
  background-color: #e9ecef;
}
.dispatch-info {
  flex: 1;
}
.dispatch-id {
  font-weight: bold;
  margin-bottom: 5px;
}
.dispatch-route {
  display: flex;
  align-items: center;
  gap: 5px;
}
.dispatch-route .start-station,
.dispatch-route .end-station {
  font-weight: bold;
}
.arrow {
  font-size: 18px;
  color: #007bff;
}
.dispatch-details {
  display: flex;
  align-items: center;
  gap: 10px;
}
.quantity {
  font-weight: bold;
  color: #28a745;
}
.status {
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}
.status.待执行 {
  background-color: #ffc107;
  color: #fff;
}
.status.已完成 {
  background-color: #28a745;
  color: #fff;
}
.dispatch-action {
  display: flex;
  align-items: center;
}
.nav-btn {
  padding: 6px 12px;
  background-color: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}
.nav-btn:hover {
  background-color: #0056b3;
}
.navigation-panel {
  position: fixed;
  top: 60px;
  right: 20px;
  width: 300px;
  max-height: calc(100vh - 80px);
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 15px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  overflow-y: auto;
  z-index: 1000;
}
.nav-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.close-nav-btn {
  background: none;
  border: none;
  font-size: 20px;
  cursor: pointer;
  color: #495057;
}
.nav-summary {
  display: flex;
  justify-content: space-between;
  margin-bottom: 15px;
}
.nav-route {
  font-weight: bold;
  font-size: 16px;
}
.nav-stats {
  display: flex;
  gap: 15px;
}
.distance,
.duration {
  font-size: 14px;
  color: #6c757d;
}
.nav-instructions {
  margin-top: 10px;
}
.instruction-list {
  max-height: 200px;
  overflow-y: auto;
}
.instruction-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 5px 0;
}
.instruction-icon {
  width: 20px;
  height: 20px;
  background-color: #007bff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}
.instruction-text {
  flex: 1;
}
.instruction-distance {
  color: #6c757d;
  font-size: 12px;
}
.route-line {
  width: 20px;
  height: 2px;
  background-color: #007bff;
  border-radius: 1px;
}
.legend {
  position: fixed;
  bottom: 20px;
  left: 20px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.legend-section {
  margin-bottom: 10px;
}
.legend-section h4 {
  margin: 0 0 5px;
  font-size: 14px;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
}
.legend-item img {
  width: 20px;
  height: 20px;
}
.legend-item .nav-marker {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  color: white;
  font-weight: bold;
}
.legend-item .start-marker {
  background-color: #28a745;
}
.legend-item .end-marker {
  background-color: #dc3545;
}
.legend-item .route-line {
  width: 20px;
  height: 2px;
  background-color: #007bff;
  border-radius: 1px;
}
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(255, 255, 255, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}
.loading-spinner {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}
.loading-spinner .spinner {
  width: 40px;
  height: 40px;
  border: 4px solid #f3f3f3;
  border-top: 4px solid #007bff;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}
.loading-spinner span {
  font-size: 16px;
  color: #495057;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.map-container {
  flex: 1;
  position: relative;
  width: 100%;
  height: calc(100vh - 120px);
}
.tooltip {
  position: absolute;
  background-color: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 5px 10px;
  border-radius: 4px;
  font-size: 12px;
  z-index: 1000;
}

</style>