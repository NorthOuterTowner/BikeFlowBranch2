<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'
import 'ol/ol.css'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM' 

const mapContainer = ref(null)
let mapInstance = null
const router = useRouter()

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
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度</h1>
        <div class="search-container">
          <input type="text" placeholder="搜索站点..." class="search-input" />
          <button class="search-button">搜索</button>
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
  box-sizing: border-box;  /* 确保padding不会增加总宽度 */
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
}

.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
}

.search-container {
  width: 40%;
  display: flex; 
}

.search-input {
  height: 30px; 
  padding: 4px 8px; 
  border-radius: 4px;
  border: 1px solid #ccc;
  flex: 1;
}

.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-left: 20px;
  gap: 8px;
}

.user-top {
  display: flex;
  align-items: center;
  gap: 40px;
}

.welcoming {
  font-size: 14px;
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
}

.search-button {
  background-color: #091275;
  color: white;
  border: none;
  padding: 6px 10px;
  border-radius: 4px;
  cursor: pointer;
}

.map-container {
  flex: 1;
  width: 100%;
  min-height: 0;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
</style>