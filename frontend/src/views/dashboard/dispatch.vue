<script setup>
import { ref, onMounted, computed, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import LineString from 'ol/geom/LineString'
import { fromLonLat } from 'ol/proj'
import { Style, Stroke, Fill, Circle, Text } from 'ol/style'
import { Zoom } from 'ol/control' // 确保导入 Zoom
import request from '@/api/axios' // Axios 请求实例
import { startDispatch,cancelDispatch, getStationAssign, getDispatch } from '../../api/axios'
import Overlay from 'ol/Overlay'

// ==================== 工具&响应式数据 ====================
const mapStatus = (statusInt) => {
  const s = Number(statusInt)
  switch (s) {
    case 0: return '待执行';
    case 1: return '正在执行';
    case 2: return '已完成';
    default: return '未知';
  }
}


const router = useRouter() // Vue Router 实例
const mapContainer = ref(null) // 地图容器的引用
const showHighlight = ref(false)  // 默认不显示右上角出度站点
let mapInstance = null // OpenLayers 地图实例
let vectorLayer = null // 用于绘制调度方案要素的矢量图层
let popupOverlay; // 保存 overlay

const welcoming = ref('管理员，欢迎您！') // 欢迎信息
const fixedDate = computed(() => {
  // 获取固定日期，优先从 localStorage 取，否则取当前日期
  return localStorage.getItem('selectedDate') || new Date().toISOString().split('T')[0]
})
const currentHour = getCurrentHourString() // 获取当前小时字符串

const now = new Date().toISOString()
const lookup_date = ref('2025-06-13')
const lookup_hour = ref(9)

// 后端调度列表
const allDispatchList = ref([])

// 当前选中的调度方案
const selectedPlan = ref(null)

// 当前状态过滤
const currentStatusFilter = ref('全部')

// 多选
const selectedIds = ref([])

// 高亮站点
const highlightStationList = ref([])

// ==================== 函数定义 ====================

/**
 * 获取当前小时字符串 (例如 "16:00")
 */
function getCurrentHourString() {
  const now = new Date()
  const hour = now.getHours().toString().padStart(2, '0')
  return `${hour}:00`
}

/**
 * 登出功能
 */
const logout = async () => {
  try {
    await request.post('/api/user/logout')
    localStorage.removeItem('token')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    router.push('/login')
  }
}

// 样式
function getStationStyle(station) {
  return new Style({
    image: new Circle({
      radius: 8,
      fill: new Fill({ color: 'rgba(66, 133, 244, 0.8)' }),
      stroke: new Stroke({ color: '#fff', width: 2 })
    }),
    text: new Text({
      text: station.name,
      font: 'bold 12px sans-serif',
      fill: new Fill({ color: 'white' }),
      stroke: new Stroke({ color: '#000', width: 2 }),
      offsetY: -15
    })
  })
}
function getRouteStyle(count) {
  return new Style({
    stroke: new Stroke({
      color: 'orange',
      width: count > 5 ? 4 : 2,
      lineDash: [5,5]
    })
  })
}

// 查看调出站点并高亮
async function highlightStations() {
  // 如果已经显示，就收起
  if (showHighlight.value) {
    showHighlight.value = false;
    highlightStationList.value = [];
    
    // 同时清除地图上的红点
    if (vectorLayer) {
      const source = vectorLayer.getSource();
      const oldHighlights = source.getFeatures().filter(f => f.get('type') === 'highlight');
      oldHighlights.forEach(f => source.removeFeature(f));
    }
    return;
  }

  // 第一次点击，获取数据并显示
  try {
    const stations = await getStationAssign({ date: lookup_date.value, hour: lookup_hour.value });
    if (!stations.length) {
      console.warn('未获取到调出站点数据');
      highlightStationList.value = [];
      return;
    }
    highlightStationList.value = stations;
    highlightStationsOnMap(stations);
    showHighlight.value = true; // 显示
  } catch (e) {
    console.error('获取调出站点失败', e);
    highlightStationList.value = [];
  }
}

function highlightStationsOnMap(stations) {
  if (!vectorLayer) return;

  // 清除原有高亮点（只清除红点，正常调度线和点不清除）
  // 或者添加到单独图层：更优雅
  const source = vectorLayer.getSource();
  
  // 假设红点 feature 有特殊属性 type: 'highlight'
  const oldHighlights = source.getFeatures().filter(f => f.get('type') === 'highlight');
  oldHighlights.forEach(f => source.removeFeature(f));

  stations.forEach(station => {
    if (!station.lng || !station.lat) return;  // 确保有经纬度
    const feature = new Feature({
      geometry: new Point(fromLonLat([station.lng, station.lat])),
      name: station.station_name,
      bikes: station.bikes_to_move,
      type: 'highlight'
    });
    feature.setStyle(new Style({
      image: new Circle({
        radius: 8,
        fill: new Fill({ color: 'red' }),
        stroke: new Stroke({ color: '#fff', width: 2 })
      })
    }));
    source.addFeature(feature);
  });
}


/**
 * 在地图上绘制选定的调度方案
 */
function drawDispatchPlanOnMap(plan) {
  console.log('绘制 item:', plan)

  const startLng = parseFloat(plan.start_station.lng)
  const startLat = parseFloat(plan.start_station.lat)
  const endLng = parseFloat(plan.end_station.lng)
  const endLat = parseFloat(plan.end_station.lat)

  console.log('绘制用的 startLng, startLat:', startLng, startLat)
  console.log('绘制用的 endLng, endLat:', endLng, endLat)

  const startCoord = fromLonLat([startLng, startLat])
  const endCoord = fromLonLat([endLng, endLat])

  console.log('绘制用的转换后 startCoord:', startCoord)
  console.log('绘制用的转换后 endCoord:', endCoord)

  if (!vectorLayer) {
    console.warn('矢量图层未初始化。')
    return
  }
  vectorLayer.getSource().clear() // 清除现有要素

  if (!plan) return

  const features = []

  filteredDispatchList.value.forEach(item => {
    // 绘制起点
    const startFeature = new Feature({
      geometry: new Point(fromLonLat([item.start_station.lng, item.start_station.lat]))
    })
    startFeature.setStyle(getStationStyle(item.start_station))
    features.push(startFeature)

    // 绘制终点
    const endFeature = new Feature({
      geometry: new Point(fromLonLat([item.end_station.lng, item.end_station.lat]))
    })
    endFeature.setStyle(getStationStyle(item.end_station))
    features.push(endFeature)

    // 绘制路线
    const lineFeature = new Feature({
      geometry: new LineString([
        fromLonLat([item.start_station.lng, item.start_station.lat]),
        fromLonLat([item.end_station.lng, item.end_station.lat])
      ])
    })
        // startFeature
    console.log('startFeature geometry:', startFeature.getGeometry().getCoordinates())

    // endFeature
    console.log('endFeature geometry:', endFeature.getGeometry().getCoordinates())

    // lineFeature
    console.log('lineFeature geometry:', lineFeature.getGeometry().getCoordinates())

    // 如果是选中的方案，线条样式更粗/颜色不同
    if (selectedPlan.value && selectedPlan.value.schedule_id === item.schedule_id) {
      lineFeature.setStyle(new Style({
        stroke: new Stroke({
          color: 'red',
          width: 6
        })
      }))
    } else {
      lineFeature.setStyle(getRouteStyle(item.bikes_to_move))
    }
    features.push(lineFeature)
  })

  vectorLayer.getSource().addFeatures(features)
  if (features.length) {
    const extent = vectorLayer.getSource().getExtent()
    mapInstance.getView().fit(extent, { padding: [50,50,50,50], duration: 1000, maxZoom: 16 })
  }

  console.log('=== 准备添加要素 ===')
  console.log('当前 view center:', mapInstance.getView().getCenter())
console.log('当前 zoom:', mapInstance.getView().getZoom())



}

/**
 * 选择调度方案并更新地图显示
 */
const selectPlan = (plan) => {
  selectedPlan.value = plan
  drawDispatchPlanOnMap(plan)

  if (plan.start_station && plan.end_station) {
    console.log('--- 选中的调度方案 ---')
    console.log('start_station:', plan.start_station)
    console.log('end_station:', plan.end_station)

    const startLng = parseFloat(plan.start_station.lng)
    const startLat = parseFloat(plan.start_station.lat)
    const endLng = parseFloat(plan.end_station.lng)
    const endLat = parseFloat(plan.end_station.lat)

    console.log('原始坐标：')
    console.log('Start:', startLng, startLat)
    console.log('End:', endLng, endLat)

    const start = fromLonLat([startLng, startLat])
    const end = fromLonLat([endLng, endLat])

    console.log('转换后的坐标：')
    console.log('Start:', start)
    console.log('End:', end)

    // 创建临时 LineString
    const tempLine = new LineString([start, end])
    const extent = tempLine.getExtent()

    console.log('计算出的 extent:', extent)

    // 也算一下简单的中心点
    const center = [
      (start[0] + end[0]) / 2,
      (start[1] + end[1]) / 2
    ]
    console.log('简单中心点:', center)

    // 用 fit
    mapInstance.getView().fit(extent, {
      padding: [100,100,100,100],
      duration: 500,
      maxZoom: 16
    })
    console.log('执行 fit 到 extent')
  } else {
    console.warn('plan.start_station 或 plan.end_station 不存在！')
  }
}

// ==================== 生命周期钩子 ====================
onMounted(async () => {
  await fetchDispatch()
  await nextTick() // 确保 DOM 渲染完成

  // 初始化地图
  mapInstance = new Map({
    target: mapContainer.value,
    layers: [
      new TileLayer({ source: new OSM() }) // OpenStreetMap 底图
    ],
    view: new View({
      center: fromLonLat([-74.0576, 40.7312]), 
      zoom: 14, // 默认缩放级别
      maxZoom: 20,
      minZoom: 3
    }),
    controls: [] // 初始不添加默认控件
  })

  // 添加自定义缩放控件
  const zoomControl = new Zoom({
    className: 'ol-zoom-custom' // 使用自定义 CSS 类
  })
  mapInstance.addControl(zoomControl)

  // 初始化矢量图层并添加到地图
  vectorLayer = new VectorLayer({
    source: new VectorSource(),
    zIndex: 10 // 确保在瓦片图层之上
  })
  mapInstance.addLayer(vectorLayer)

  // 默认选中第一个调度方案并绘制
  if (allDispatchList.value.length) selectPlan(allDispatchList.value[0])

  console.log('调度详情页面地图初始化完成。')
  console.log('allDispatchList:', JSON.stringify(allDispatchList.value, null, 2))

  // 创建 overlay（tooltip）
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  tooltip.style.background = 'rgba(0,0,0,0.7)';
  tooltip.style.color = '#fff';
  tooltip.style.padding = '2px 6px';
  tooltip.style.borderRadius = '4px';
  tooltip.style.fontSize = '12px';
  tooltip.style.whiteSpace = 'nowrap';

  popupOverlay = new Overlay({
    element: tooltip,
    offset: [10, 0],
    positioning: 'center-left',
    stopEvent: false
  });
  mapInstance.addOverlay(popupOverlay);

  // 鼠标移动事件
  mapInstance.on('pointermove', e => {
    const pixel = mapInstance.getEventPixel(e.originalEvent);
    const feature = mapInstance.forEachFeatureAtPixel(pixel, f => f);
    if (feature && feature.get('type') === 'highlight') {
      const name = feature.get('name');
      const bikes = feature.get('bikes') ?? '-';
      tooltip.innerHTML = `${name}：${bikes}辆`;
      popupOverlay.setPosition(e.coordinate);
      tooltip.style.display = 'block';
    } else {
      tooltip.style.display = 'none';
    }
  });
})

// ==================== 获取数据 ====================
async function fetchDispatch() {
  try {
    const res = await getDispatch('2025-06-13T09:00:00Z')
    console.log('后端返回数据：', res.data)

    if (!res.data || typeof res.data !== 'object') {
      console.error('接口返回不是 JSON：', res.data)
      return
    }

    const schedules = res.data?.schedules || res.data?.data || res.data || []

    if (!Array.isArray(schedules)) {
      console.error('返回的调度列表不是数组：', schedules)
      return
    }
    console.log('调度列表:', schedules)
    console.log(schedules.map(s => ({
  startName: s.startStation?.station_name,
  endName: s.endStation?.station_name
})));

    allDispatchList.value = schedules.map(item => ({
      ...item,
      //start_station_name: item.start_station?.name || '',
      //end_station_name: item.end_station?.name || '',
      statusInt: item.status,
      //status: mapStatus(Number(item.status))
    }))
  } catch (e) {
    console.error('获取调度数据失败', e)
  }
}


// ==================== 根据状态过滤 ====================
// 计算属性，映射状态文字并过滤
const filteredDispatchList = computed(() => {
  const filtered = allDispatchList.value.filter(item => {
    if (currentStatusFilter.value === '全部') return true;
    return item.statusInt === currentStatusFilter.value;
  });

     // 如果过滤的是“全部”，才排序
  if (currentStatusFilter.value === '全部') {
    return filtered.slice().sort((a, b) => {
      // mapStatus 返回状态文字
      const statusA = a.statusInt;
      const statusB = b.statusInt;

      // 把“已完成”的排后面
      if (statusA === '已完成' && statusB !== '已完成') return 1; // a排后面
      if (statusA !== '已完成' && statusB === '已完成') return -1; // b排后面

      // 其它状态保持原顺序
      return 0;
    });
  }

  // 如果过滤了具体状态，直接返回
  return filtered;
  })


// ==================== adopt/撤销 ====================
async function handleStart(item) {
  try {
    await startDispatch({
      startStation: item.start_station.name,
      endStation: item.end_station.name,
      number: item.bikes_to_move,
      dispatchDate: lookup_date.value,
      dispatchHour: lookup_hour.value,
      dispatchId: item.schedule_id
    })
    item.statusInt = "正在执行"
    item.status = mapStatus(item.statusInt)
  } catch (e) {
    console.error('执行调度失败', e)
  }
}

async function handleCancel(item) {
  try {
    await cancelDispatch({
      startStation: item.start_station.name,
      endStation: item.end_station.name,
      number: item.bikes_to_move,
      dispatchDate: lookup_date.value,
      dispatchHour: lookup_hour.value,
      dispatchId: item.schedule_id
    })
    item.statusInt = "待执行"
    item.status = mapStatus(item.statusInt)
  } catch (e) {
    console.error('撤销调度失败', e)
  }
}

// 批量
async function batchStart() {
  const items = filteredDispatchList.value.filter(i => selectedIds.value.includes(i.schedule_id))
  for (const item of items) await handleStart(item)
}
async function batchCancel() {
  const items = filteredDispatchList.value.filter(i => selectedIds.value.includes(i.schedule_id))
  for (const item of items) await handleCancel(item)
}

</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度详情</h1>
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

    <div class="main-content">
  <!-- 左侧列表面板 -->
  <div class="dispatch-list-panel">
    <!-- 状态切换 -->
    <div class="status-buttons">
      <button
        v-for="status in ['全部','待执行','正在执行','已完成']"
        :key="status"
        @click="currentStatusFilter = status"
        :class="['status-btn', currentStatusFilter === status ? 'active' : '']"
      >
        {{ status }}
      </button>
    </div>
    <!-- 批量操作 -->
    <div class="button-row">
      <div class="batch-buttons">
        <button @click="batchStart">批量采用</button>
        <button @click="batchCancel">批量撤销</button>
      </div>

      <div class="highlight-btn">
        <button @click="highlightStations">
          {{ showHighlight ? '隐藏调出站点' : '查看调出站点' }}
        </button>

      </div>
    </div>


    <!-- 表格列表 -->
    <table class="plan-table">
  <thead>
    <tr>
      <th>多选</th>
      <th>起止站点</th>
      <th>状态</th>
      <th>数量</th>
      <th>操作</th>
    </tr>
    
  </thead>
  
  <tbody>
    <tr
      v-for="item in filteredDispatchList"
      :key="item.schedule_id"
      :class="{ selected: selectedPlan && selectedPlan.schedule_id === item.schedule_id }"
      @click="selectPlan(item)"
    >
      <td>
        <input type="checkbox" v-model="selectedIds" :value="item.schedule_id" @click.stop />
      </td>
      <td>{{ item.start_station.name }} → {{ item.end_station.name }}</td>

      <!-- 状态：加圆角彩色标签 -->
      <td>
        <span 
          :class="[
            'status-tag',
            item.statusInt === '待执行' ? 'status-pending' : '',
            item.statusInt === '正在执行' ? 'status-running' : '',
            item.statusInt === '已完成' ? 'status-finished' : ''
          ]"
        >
          {{ item.statusInt }}
        </span>
      </td>

      <td>{{ item.bikes_to_move ?? '-' }}</td>

      <!-- 操作按钮判断用 mapStatus 转文字 -->
      <td>
        <button v-if="item.statusInt === '待执行'" @click.stop="handleStart(item)">采用</button>
        <button v-if="item.statusInt === '正在执行'" @click.stop="handleCancel(item)">撤销</button>
      </td>
    </tr>
  </tbody>
</table>



    
  </div>

  <!-- 右侧地图面板 -->
  <div class="map-panel">
    <div ref="mapContainer" class="map"></div>
  </div>
  <div class="highlight-info-panel" v-if="showHighlight && highlightStationList.length">
    <h4>调出站点</h4>
    <ul>
      <li v-for="station in highlightStationList" :key="station.station_name">
        {{ station.station_name }}
      </li>
    </ul>
  </div>

</div>
  </div>


</template>

<style scoped>
/* header 和布局样式 */
.title { font-size:18px; font-weight:bold; }
.user-info { text-align:right; }
.logout-button { background:#091275; color:white; border:none; padding:4px 8px; border-radius:4px; cursor:pointer; }
/* 左侧面板 */
.plan-item { border:1px solid #ddd; margin-bottom:6px; padding:6px; border-radius:4px; cursor:pointer; }
.plan-item.selected { background:#e6f0ff; border-color:#091275; }
.plan-header { display:flex; justify-content:space-between; }
.plan-status { font-size:12px; }
.plan-actions button { margin-right:4px; white-space: nowrap;}

/* 批量按钮 */
.batch-buttons, .highlight-btn { margin-top:8px; }
.batch-buttons button, .highlight-btn button { margin-right:4px; }

.app-container {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0;
}

.main-content {
  flex: 1; /* 剩余高度全部占满 */
  display: flex;
  overflow: hidden; /* 防止溢出 */
}

.dispatch-list-panel {
  width: 400px; /* 固定宽度 */
  overflow-y: auto;
  border-right: 1px solid #ccc;
  padding: 10px;
  box-sizing: border-box;
  background-color: #fafafa;
}

.map-panel {
  flex: 1; /* 占据剩余宽度 */
  position: relative;
}

.map {
  position: absolute;
  top: 0; bottom: 0; left: 0; right: 0;
  width: 100%;
  height: 100%;
}
.map-panel :deep(.ol-zoom-custom) {
  position: absolute;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
}

.plan-table {
  width: 100%;
  border-collapse: collapse; /* 合并边框 */
  font-size: 14px;
  margin-bottom: 12px;
}

.plan-table th,
.plan-table td {
  border: 1px solid #ccc;   /* 加边框 */
  padding: 8px 12px;
  text-align: left;
  vertical-align: middle;
  background-color: #fff;   /* 白底 */
  white-space: nowrap; /* 防止换行 */
}

.plan-table th {
  background-color: #f8f8f8;   /* 浅灰背景 */
  color: #333;                /* 深灰字体 */
  font-weight: 600;
  text-align: center;         /* 居中对齐 */
  padding: 8px 12px;
  border-bottom: 1px solid #ddd;
  white-space: nowrap; /* 防止换行 */
}

.plan-table tr:hover {
  background-color: #e8f0fe; /* 鼠标悬浮高亮 */
}

.plan-table tr.selected {
  background-color: #c7dbff; /* 选中行高亮 */
  border-color: #4169e1;
}

.status-buttons {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}

.status-btn {
  padding: 6px 14px;
  border: 1px solid #007bff;
  background-color: #f8f9fa;
  color: #007bff;
  cursor: pointer;
  border-radius: 4px;
  font-weight: 500;
  transition: background-color 0.3s ease;
  white-space: nowrap;
}

.status-btn.active,
.status-btn:hover {
  background-color: #007bff;
  color: white;
}
.highlight-btn {
  margin-top: 10px;
}
.status-tag {
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 600;       /* 更加粗 */
  border-radius: 12px;
  color: #fff;
  text-align: center;     /* 水平居中 */
  vertical-align: middle; /* 垂直对齐中线 */
  white-space: nowrap;
}


/* 待执行：黄色背景 */
.status-pending {
  background-color: #f0ad4e;
}

/* 正在执行：红色背景 */
.status-running {
  background-color: #d9534f;
}

/* 已完成：绿色背景 */
.status-finished {
  background-color: #5cb85c;
}

.plan-table button,
.batch-buttons button,
.highlight-btn button {
  padding: 4px 10px;
  margin: 2px;
  border: none;
  border-radius: 6px;
  background-color: #409eff;     /* 默认蓝色背景 */
  color: #fff;
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.2s;
  white-space: nowrap;
}

.plan-table button:hover,
.batch-buttons button:hover,
.highlight-btn button:hover {
  background-color: #66b1ff;     /* 浅蓝 hover */
}

.button-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.highlight-info-panel {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 220px;
  max-height: 300px;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.95);
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 13px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  z-index: 1000;
}
.highlight-info-panel h4 {
  margin: 0 0 6px;
  font-size: 14px;
  font-weight: 600;
}
.highlight-info-panel ul {
  margin: 0;
  padding-left: 16px;
}
.highlight-info-panel li {
  line-height: 1.5;
}
.highlight-info-panel li::before {
  content: '•';
  color: #409eff; /* 蓝色点 */
  font-weight: bold;
  display: inline-block;
  width: 1em; /* 确保点和文本对齐 */
  margin-left: -1em; /* 向左移动点 */
}

</style>