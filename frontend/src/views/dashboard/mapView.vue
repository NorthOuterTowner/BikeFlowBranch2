<script setup>
import { ref, onMounted } from 'vue'
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
const router = useRouter()
const stations = ref([])

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
  const bikeCount = station.bikeCount || 0 // 假设stations数据中有bikeCount字段
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

function getNowDatetimeLocal() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hours = String(now.getHours()).padStart(2, '0')
  const minutes = String(now.getMinutes()).padStart(2, '0')

  // datetime-local 格式是：YYYY-MM-DDTHH:mm
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

const dateTime = ref(getNowDatetimeLocal())
const welcoming = ref('管理员，欢迎您！')
const searchQuery = ref('')  // 添加搜索查询的响应式数据

onMounted(() => {
  mapInstance = new Map({
    target: mapContainer.value,
    layers: [
      new TileLayer({
        source: new OSM(), // OpenStreetMap 图层
      }),
    ],
    view: new View({
      center: [-9755221.54, 5141303.88],
      zoom: 11,
      maxZoom: 20,
      minZoom: 3
    }),
  })

  // 构建矢量图层
  const features = stations.map(station => {
    const feature = new Feature({
      geometry: new Point(fromLonLat([station.longitude, station.latitude]))
    })
    
    // 设置站点样式
    feature.setStyle(getStationStyle(station))
    
    // 将站点数据存储到feature中，便于后续使用
    feature.set('stationData', station)
    
    return feature
  })

  const vectorLayer = new VectorLayer({
    source: new VectorSource({
      features: features
    })
  })

  mapInstance.addLayer(vectorLayer)
})

const logout = async () => {
  try {
    await axios.post('/api/user/logout') // 后端登出接口
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  }

  // 跳转到登录页
  router.push('/login')
}

const handleSearch = () => {
  if (searchQuery.value.trim()) {
    console.log('搜索:', searchQuery.value)
    // 在这里添加搜索逻辑
    // 例如：调用API搜索站点，或者在地图上高亮显示
  }
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
  width: 220px;
}

.datetime-input {
  width: 100%;
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid #ccc;
  cursor: pointer;
  box-sizing: border-box;
}

.logout-button {
  margin-left: 10px;
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap; 
}

.search-button {
  background-color: #091275;
  color: white;
  border: none;
  padding: 6px 10px;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
}

.legend {
  position: absolute;
  top: 120px;
  right: 20px;
  background-color: rgba(255, 255, 255, 0.9);
  padding: 10px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  z-index: 10; /* 降低 z-index 值 */
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

.map-container {
  flex: 1;
  width: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  position: relative;
  z-index: 1; /* 为地图容器设置较低的 z-index */
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