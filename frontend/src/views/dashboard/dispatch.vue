<script setup>
import { ref, onMounted, computed, nextTick ,onBeforeUnmount} from 'vue'
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
import { Style, Stroke, Fill, Circle, Text, Icon } from 'ol/style'
import { Zoom } from 'ol/control' // 确保导入 Zoom
import request from '@/api/axios' // Axios 请求实例
import { startDispatch,cancelDispatch, getStationAssign, getDispatch, rejectDispatch, getDispatchPlan } from '../../api/axios'
import Overlay from 'ol/Overlay'
import { ElMessage } from 'element-plus'

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
// 站点名称映射函数 - 根据站点ID获取显示名称(假的...)
const getStationDisplayName = (station) => {
  if (!station) return '未知站点'
  
  // 优先使用 name 字段
  if (station.name) return station.name
  
  // 其次使用 station_name 字段
  if (station.station_name) return station.station_name
  
  // 最后使用 id 作为显示名称
  return station.id || '未知站点'
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

//const now = new Date().toISOString()
const lookup_date = ref('2025-06-13')
const lookup_hour = ref(9)

// 2. 在响应式数据部分添加导航相关变量
const navigationActive = ref(false)
const currentRoute = ref(null)
const navigationInstructions = ref([])
const routeDistance = ref(0)
const routeDuration = ref(0)
const selectedDispatch = ref(null)
const loading = ref(false)
let navigationLayer = null

const topPanelHeight = ref(185) // 初始高度
const minHeight = 100
const maxHeight = 600
let isResizing = false

function startResize(event) {
  isResizing = true
  document.addEventListener('mousemove', handleResize)
  document.addEventListener('mouseup', stopResize)
}

function handleResize(event) {
  if (!isResizing) return
  // 获取当前鼠标位置距离页面顶部的高度
  const newHeight = event.clientY - document.querySelector('.app-header').offsetHeight
  // 限制最大最小高度
  if (newHeight > minHeight && newHeight < maxHeight) {
    topPanelHeight.value = newHeight
  }
}

function stopResize() {
  isResizing = false
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', stopResize)
}

// 清理事件监听
onBeforeUnmount(() => {
  document.removeEventListener('mousemove', handleResize)
  document.removeEventListener('mouseup', stopResize)
})

// 3. 修改 onMounted 中的地图初始化，添加导航图层
// 在 vectorLayer 初始化后添加：
navigationLayer = new VectorLayer({
  source: new VectorSource(),
  zIndex: 15 // 确保在其他图层之上
})

/**
 * 调用 OpenRouteService 获取路线
 */
async function getRoute(startCoord, endCoord) { 
  try {
    const response = await fetch('/guide/route', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ startCoord, endCoord })
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.error || '路线请求失败')
    }

    const data = await response.json()
    console.log('后端代理返回路线数据:', data)
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
    const startCoord = [parseFloat(dispatch.start_station.lng), parseFloat(dispatch.start_station.lat)]
    const endCoord = [parseFloat(dispatch.end_station.lng), parseFloat(dispatch.end_station.lat)]
    
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
    
    // 清除现有图层，只保留导航相关内容
    vectorLayer.getSource().clear()
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
      image: new Circle({
        radius: 12,
        fill: new Fill({ color: '#28a745' }),
        stroke: new Stroke({ color: '#fff', width: 2 })
      }),
      text: new Text({
        text: '起',
        font: 'bold 12px sans-serif',
        fill: new Fill({ color: 'white' })
      })
    }))
    
    // 添加终点标记
    const endMarker = new Feature({
      geometry: new Point(fromLonLat(endCoord))
    })
    endMarker.setStyle(new Style({
      image: new Circle({
        radius: 12,
        fill: new Fill({ color: '#dc3545' }),
        stroke: new Stroke({ color: '#fff', width: 2 })
      }),
      text: new Text({
        text: '终',
        font: 'bold 12px sans-serif',
        fill: new Fill({ color: 'white' })
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
  
  // 恢复原来的调度方案显示
  if (selectedPlan.value) {
    drawDispatchPlanOnMap(selectedPlan.value)
  }
}


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
  const confirmed = window.confirm('确定要退出登录吗？')
  if (!confirmed) {
    // 用户取消退出
    return
  }

  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    sessionStorage.clear()
    router.push('/login')
  }
}

// 站点样式
function getStationStyle(station) {
  let iconSrc = '/icons/BlueLocationRound.svg'

  return new Style({
    image: new Icon({
      src: iconSrc,
      scale: 1.5,
      anchor: [0.5, 1]
    }),
    text: new Text({
      text: station.name,
      font: 'bold 12px sans-serif',
      fill: new Fill({ color: 'white' }),
      stroke: new Stroke({ color: '#000', width: 2 }),
      offsetY: -40
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
    console.log('拿到的站点：', JSON.stringify(stations, null, 2))

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
// 全局或模块作用域变量
let animating = false;
let animationStart = null;
let arrowFeatures = [];
let currentLine = null;
const arrowCount = 5;    // 箭头数量
const animationDuration = 4000; // 毫秒

function drawDispatchPlanOnMap(plan) {
  console.log('绘制 item:', plan);

  if (!vectorLayer || !mapInstance) {
    console.warn('矢量图层未初始化。');
    return;
  }

  // 清除旧图层内容与动画
  vectorLayer.getSource().clear();
  mapInstance.un('postrender', animateArrow);
  animating = false;
  animationStart = null;
  arrowFeatures = [];
  currentLine = null;

  if (!plan) return;

  const features = [];

  const item = plan;

  const startPt = fromLonLat([item.start_station.lng, item.start_station.lat]);
  const endPt = fromLonLat([item.end_station.lng, item.end_station.lat]);



  // 路线
  const line = new LineString([startPt, endPt]);
  const lineFeature = new Feature({ geometry: line });
  lineFeature.setStyle(new Style({
    stroke: new Stroke({
      color: 'blue',
      width: 4
    })
  }));
  features.push(lineFeature);

  // 箭头 Features
  for (let i = 0; i < arrowCount; i++) {
    const arrow = new Feature({ geometry: new Point(startPt) });
    arrow.offsetIndex = i;
    arrowFeatures.push(arrow);
    features.push(arrow);
  }

  // 起点
  const startFeature = new Feature({ geometry: new Point(startPt) });
  startFeature.setStyle(getStationStyle(item.start_station));
  features.push(startFeature);

  // 终点
  const endFeature = new Feature({ geometry: new Point(endPt) });
  endFeature.setStyle(getStationStyle(item.end_station));
  features.push(endFeature);

  vectorLayer.getSource().addFeatures(features);

  // 自动缩放地图
  const extent = vectorLayer.getSource().getExtent();
  if (extent) {
    mapInstance.getView().fit(extent, { padding: [50, 50, 50, 50], duration: 1000, maxZoom: 16 });
  }

  // 启动动画
  currentLine = line;
  animating = true;
  animationStart = null;
  mapInstance.on('postrender', animateArrow);
  mapInstance.render();

  // 返回当前选中线路 Feature（可选）
  return lineFeature;
}


// 用于计算箭头样式（icon + 方向）
function getArrowStyle(rotation) {
  return new Style({
    image: new Icon({
      src: '/icons/arrow.png', // 替换成你项目里的箭头图路径
      scale: 0.07,
      rotation: rotation,
      rotateWithView: true
    })
  });
}


function animateArrow(event) {
  if (!animating || !currentLine || arrowFeatures.length === 0) return;

  const vectorContext = event.vectorContext;
  const frameState = event.frameState;

  if (!animationStart) animationStart = frameState.time;
  const elapsed = frameState.time - animationStart;
  const fraction = (elapsed % animationDuration) / animationDuration;
  const spacing = 1 / arrowCount;

  const coords = currentLine.getCoordinates();
  const segmentCount = coords.length - 1;
  if (segmentCount <= 0) return;

  arrowFeatures.forEach((arrow, idx) => {
    if (!arrow || typeof arrow.getGeometry !== 'function' || typeof arrow.setStyle !== 'function') {
      console.error(`❌ 无效的 arrowFeature[${idx}]`, arrow);
      return;
    }

    const offset = (fraction + arrow.offsetIndex * spacing) % 1;
    const segmentIndex = Math.floor(offset * segmentCount) % segmentCount;

    const [x1, y1] = coords[segmentIndex];
    const [x2, y2] = coords[segmentIndex + 1];

    // 插值计算箭头当前坐标
    const localFraction = (offset * segmentCount) % 1;
    const coord = [
      x1 + (x2 - x1) * localFraction,
      y1 + (y2 - y1) * localFraction
    ];

    // 使用屏幕像素坐标计算方向
    const pixel1 = mapInstance.getPixelFromCoordinate([x1, y1]);
    const pixel2 = mapInstance.getPixelFromCoordinate([x2, y2]);
    const rotation = Math.atan2(pixel2[1] - pixel1[1], pixel2[0] - pixel1[0]);

    try {
      arrow.getGeometry().setCoordinates(coord);
      const style = getArrowStyle(rotation);
      if (!style) {
        console.error(`❌ getArrowStyle(${rotation}) 返回空`);
        return;
      }
      arrow.setStyle(style);
    } catch (e) {
      console.error(`❌ 渲染 arrow[${idx}] 出错:`, e);
    }
  });

  mapInstance.render(); // 保持动画
}



/**
 * 选择调度方案并更新地图显示
 */
const selectPlan = (plan) => {
  selectedPlan.value = plan
  if (navigationLayer) {
    navigationLayer.getSource().clear()
  }
  const newLineFeature = drawDispatchPlanOnMap(plan);

  if (plan.start_station && plan.end_station) {
    const startLng = parseFloat(plan.start_station.lng)
    const startLat = parseFloat(plan.start_station.lat)
    const endLng = parseFloat(plan.end_station.lng)
    const endLat = parseFloat(plan.end_station.lat)

    const start = fromLonLat([startLng, startLat])
    const end = fromLonLat([endLng, endLat])

    // 创建临时 LineString
    const tempLine = new LineString([start, end])
    const extent = tempLine.getExtent()

    // 确保新线条存在后再做 fit
    if (newLineFeature && newLineFeature.getGeometry()) {
      const extent = newLineFeature.getGeometry().getExtent();
      mapInstance.getView().fit(extent, {
        padding: [100, 100, 100, 100],
        duration: 500,
        maxZoom: 16
      });
    }
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
  mapInstance.addLayer(navigationLayer)

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
      start_station_name: getStationDisplayName(item.start_station),
      end_station_name: getStationDisplayName(item.end_station),
      statusInt: item.status,
      status: mapStatus(item.status)
    }))
    console.log('处理后的调度列表:', allDispatchList.value)
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
    const res = await startDispatch({
      startStation: item.start_station.id,
      endStation: item.end_station.id,
      number: item.bikes_to_move,
      dispatchDate: lookup_date.value,
      dispatchHour: lookup_hour.value,
      dispatchId: item.schedule_id
    });

    item.statusInt = "正在执行";
    item.status = mapStatus(item.statusInt);
    item.dispatchTime = res.data.time;

    console.log(`调度执行耗时：${res.data.time} ms`);

    // 保存当前操作的调度ID（或生成唯一标识）
    const currentScheduleId = item.schedule_id;

    const expectedDuration = res.data.time || 5000;

    setTimeout(async () => {
      console.log("开始调度接口返回时间:", res.data.time);

      // 重新从列表中获取当前 item（防止状态已被取消）
      const currentItem = allDispatchList.value.find(i => i.schedule_id === currentScheduleId);

      // 如果不存在、或状态已被撤销/非“正在执行”，跳过刷新
      if (!currentItem || currentItem.statusInt !== "正在执行") {
        console.log("调度已被撤销或状态已变，跳过刷新");
        return;
      }

      console.log("调度预计完成，开始刷新列表");
      await refreshDispatchList();

    }, expectedDuration + 1000);

  } catch (e) {
    console.error("执行调度失败", e);
  }
}

async function handleReject(item) {
  try {
    const res = await rejectDispatch({ dispatchId: item.schedule_id })
    if (res.data?.code === 200) {
      allDispatchList.value = allDispatchList.value.filter(
        i => i.schedule_id !== item.schedule_id
      )
      if (selectedPlan && selectedPlan.schedule_id === item.schedule_id) {
        selectedPlan = null
      }
      // 可选：ElMessage.success('已拒绝该调度')
    } else {
      console.error('拒绝失败：', res.data?.msg)
      // 可选：ElMessage.error(res.data?.msg || '未知错误')
    }
  } catch (e) {
    console.error('请求拒绝调度失败', e)
  }
}


async function handleCancel(item) {
  if (item.isCancelling) {
    ElMessage.success("撤销中，禁止重复操作");
    return;
  }
  item.isCancelling = true;
  ElMessage.info("正在撤销调度，请稍候...");
  try {
    const res = await cancelDispatch({
      startStation: item.start_station.id,
      endStation: item.end_station.id,
      number: item.bikes_to_move,
      dispatchDate: lookup_date.value,
      dispatchHour: lookup_hour.value,
      dispatchId: item.schedule_id
    })
    const time = res.data.time;
    const waitTime = (typeof time === 'number' && time > 0) ? time : 5000;

    // 撤销请求耗时立即显示
    item.dispatchTime = waitTime;
    console.log(`撤销调度耗时：${time} ms`);

    // 等待撤销耗时结束
    await new Promise(resolve => setTimeout(resolve, waitTime + 1000));

    await refreshDispatchList();
    // 撤销完成，状态变回待执行，耗时显示“-”
    const currentItem = allDispatchList.value.find(i => i.schedule_id === item.schedule_id);
    if (currentItem) {
      currentItem.statusInt = "待执行";
      currentItem.status = mapStatus(currentItem.statusInt);
      currentItem.dispatchTime = "";
    }
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
async function batchReject() {
  const items = filteredDispatchList.value.filter(i => 
    selectedIds.value.includes(i.schedule_id)
  )
  let successCount = 0
  for (const item of items) {
    try {
      await handleReject(item)
      successCount++
    } catch (e) {
      console.error('批量拒绝单个失败', e)
    }
  }
  ElMessage.success(`批量拒绝完成，成功：${successCount} 条`)
}

function focusStationOnMap(station) {
  console.log('点击了高亮站点：', station);

  // Check if mapInstance is initialized and station has coordinates
  if (mapInstance && station.lat && station.lng) {
    const coords = fromLonLat([station.lng, station.lat]); // Convert Lon/Lat to map projection
    console.log('将地图中心设置到坐标：', coords);

    mapInstance.getView().animate({
      center: coords,
      zoom: 16, // You can adjust the zoom level here
      duration: 500 // Animation duration in milliseconds
    });
  } else {
    console.warn('地图未初始化或站点缺少经纬度信息。', { mapInstance, station });
  }
}

async function handleUpdate() {
  try {
    const res = await getDispatchPlan(lookup_date.value, lookup_hour.value);
    const data = res.data;

    if (res.status === 200 && data.success) {
      ElMessage.success('调度方案已更新');
      await refreshDispatchList();
    } else {
      throw new Error(data.message || '调度失败');
    }
  } catch (err) {
    console.error('更新失败', err);
    ElMessage.error(err.message || '更新调度方案失败');
  }
}
async function refreshDispatchList() {
  try {
    await fetchDispatch();
    ElMessage.success('调度列表已刷新');
  } catch (e) {
    console.error('刷新调度列表失败', e);
    ElMessage.error('刷新调度列表失败，请稍后再试');
  }
}

</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度详情</h1>
        <button class="update-button" @click="handleUpdate">更新调度方案</button>
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
      <div class="top-panel" :style="{ height: topPanelHeight + 'px' }">
  <div class="top-panel-content">
    <!-- 左侧：两列按钮区 -->
    <div class="buttons-area">
      <div class="buttons-columns">
        <div class="left-column">
          <button @click="batchStart">批量采用</button>
          <button @click="batchCancel">批量撤销</button>
          <button @click="batchReject">批量拒绝</button>
          <button @click="highlightStations">
            {{ showHighlight ? '隐藏调出站点' : '查看调出站点' }}
          </button>
        </div>
        <div class="right-column">
          <button
            v-for="status in ['全部', '待执行', '正在执行', '已完成']"
            :key="status"
            @click="currentStatusFilter = status"
            :class="[
              'status-btn',
              status === '待执行' ? 'status-pending' :
              status === '正在执行' ? 'status-running' :
              status === '已完成' ? 'status-finished' : '',
              currentStatusFilter === status ? 'active' : ''
            ]"
          >
            {{ status }}
          </button>
        </div>
      </div>
    </div>

    <!-- 右侧：可滚动列表 -->
    <div class="dispatch-list-scrollable">
      <table class="plan-table">
        <thead>
          <tr>
            <th class="col-checkbox">多选</th>
            <th class="col-stations">起止站点</th>
            <th class="col-status">状态</th>
            <th class="col-number">数量</th>
            <th class="col-action">操作</th>
            <th class="col-time">耗时</th>
            <th class="col-nav">导航</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="item in filteredDispatchList"
            :key="item.schedule_id"
            :class="{ selected: selectedPlan && selectedPlan.schedule_id === item.schedule_id }"
            @click="selectPlan(item)"
          >
            <td class="col-checkbox">
              <input type="checkbox" v-model="selectedIds" :value="item.schedule_id" @click.stop />
            </td>
            <td class="col-stations">
              <div class="station-line">
                <div class="station-badge start-station">{{ item.start_station.name }}</div>
                <div class="arrow">→</div>
                <div class="station-badge end-station">{{ item.end_station.name }}</div>
              </div>
            </td>
            <td class="col-status">
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
            <td class="col-number">{{ item.bikes_to_move ?? '-' }}</td>
            <td class="col-action">
              <button
                v-if="item.statusInt === '待执行'"
                @click.stop="handleStart(item)"
                class="btn-adopt"
              >
                采用
              </button>
              <button
                v-if="item.statusInt === '待执行'"
                @click.stop="handleReject(item)"
                class="btn-reject"
              >
                拒绝
              </button>
              <button
                v-if="item.statusInt === '正在执行'"
                @click.stop="handleCancel(item)"
                class="btn-cancel"
              >
                撤销
              </button>
            </td>
            <td class="col-time">
              <span v-if="item.dispatchTime">
                {{ (item.dispatchTime / 1000).toFixed(2) }} 秒
              </span>
              <span v-else>-</span>
            </td>
            <td class="col-nav">
              <button 
                @click.stop="showNavigation(item)" 
                :disabled="loading"
                class="btn-nav"
              >
                {{ loading ? '加载中...' : '导航' }}
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

      <div class="resizer" @mousedown="startResize"></div>

      <div class="map-panel">
        <div ref="mapContainer" class="map"></div>
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
        <div class="highlight-info-panel" v-if="showHighlight && highlightStationList.length">
      <h4>调出站点</h4>
      <ul>
        <li
          v-for="station in highlightStationList"
          :key="station.station_name"
          @click="focusStationOnMap(station)"
          style="cursor: pointer;"
        >
          {{ station.station_name }}
        </li>
      </ul>
    </div>

        <div v-if="navigationActive" class="navigation-info-panel">
          <div class="navigation-header">
            <h3>导航信息</h3>
            <button class="close-btn" @click="clearNavigation">×</button>
          </div>

          <div class="navigation-content">
            <div class="route-summary">
              <div class="route-info">
                <span class="label">起点：</span>
                <span>{{ selectedDispatch?.start_station.name }}</span>
              </div>
              <div class="route-info">
                <span class="label">终点：</span>
                <span>{{ selectedDispatch?.end_station.name }}</span>
              </div>
              <div class="route-info">
                <span class="label">距离：</span>
                <span>{{ routeDistance }} 公里</span>
              </div>
              <div class="route-info">
                <span class="label">预计时间：</span>
                <span>{{ routeDuration }} 分钟</span>
              </div>
            </div>

            <div class="navigation-instructions">
              <h4>导航指令</h4>
              <div class="instructions-list">
                <div
                  v-for="(instruction, index) in navigationInstructions"
                  :key="index"
                  class="instruction-item"
                >
                  <div class="instruction-distance">{{ (instruction.distance / 1000).toFixed(2) }}km</div>
                  <div class="instruction-text">{{ instruction.instruction }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* header 和布局样式 */

.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
}

.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-left: 20px;
  gap: 15px;
  flex-shrink: 0;
}

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

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0;
}

/* 左侧按钮区 */
.left-buttons-panel {
  width: 180px;
  min-width: 150px;
  background-color: #f0f2f5;
  padding: 10px;
  box-sizing: border-box;
  border-right: 1px solid #ccc;
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
  padding: 6px 12px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin: 2px;
  background: #f0f0f0;  /* 默认背景 */
  color: #333;
  transition: background-color 0.2s, transform 0.1s;
}

.status-btn.active,
.status-btn:hover {
  /* background-color: #007bff;
  color: white; */
  transform: translateY(-1px);
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
  color: #fff;
}

/* 正在执行：红色背景 */
.status-running {
  background-color: #d9534f;
  color: #fff;
}

/* 已完成：绿色背景 */
.status-finished {
  background-color: #5cb85c;
  color: #fff;
}

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

/* .plan-table button:hover, */

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
  top: 130px;
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

/* 添加到 <style scoped> 中 */

/* 导航信息面板样式 */
.navigation-info-panel {
  position: absolute;
  top: 10px;
  right: 10px;
  width: 320px;
  max-height: 80vh;
  background: white;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  overflow: hidden;
}

.navigation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background-color: #007bff;
  color: white;
}

.navigation-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 20px;
  cursor: pointer;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background-color 0.2s;
}

.close-btn:hover {
  background-color: rgba(255, 255, 255, 0.2);
}

.navigation-content {
  padding: 16px;
  max-height: calc(80vh - 60px);
  overflow-y: auto;
}

.route-summary {
  margin-bottom: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid #eee;
}

.route-info {
  display: flex;
  flex-direction: column;  /* 改为垂直布局 */
  margin-bottom: 8px;
  font-size: 14px;
}


.route-info .label {
  font-weight: 600;
  color: #333;
}

.navigation-instructions h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.instructions-list {
  max-height: 300px;
  overflow-y: auto;
}

.instruction-item {
  display: flex;
  align-items: flex-start;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
  font-size: 13px;
}

.instruction-item:last-child {
  border-bottom: none;
}

.instruction-distance {
  min-width: 50px;
  font-weight: 600;
  color: #007bff;
  margin-right: 12px;
}

.instruction-text {
  flex: 1;
  line-height: 1.4;
  color: #666;
}

/* 导航按钮样式 */
.plan-table button[disabled] {
  background-color: #409eff;
  cursor: not-allowed;
}

.plan-table button[disabled]:hover {
  background-color: #ccc;
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
.station-line {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 6px;
}

.station-badge {
  background-color: #f0f4ff;    /* 浅底色 */
  padding: 2px 8px;
  border-radius: 6px;
  font-weight: 500;
  color: #333;
  white-space: nowrap;
}

.start-station {
  color: #2c7be5;               /* 蓝色文字 */
}

.end-station {
  color: #28a745;               /* 绿色文字 */
}

.arrow {
  color: #999;
  margin: 2px 0;
}
/* 设置各列宽度 */
.col-checkbox {
  width: 1px;
  text-align: center;
}

.col-status {
  width: 60px;
  text-align: center;
}

.col-number {
  width: 0px;
  text-align: center;
}

.col-action {
  width: 70px;
  text-align: center;
}
.col-time {
  width: 80px;
  text-align: center;
}

.col-nav {
  width: 60px;
  text-align: center;
}

/* 起止站点列自动占满剩余空间 */
.col-stations {
  min-width: 120px;
}
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.top-panel {
  overflow: hidden;
  background: #fafafa;
  border-bottom: 1px solid #ddd;
}
.buttons-columns {
  display: flex;
  flex-direction: row;
  gap: 8px;
}

.resizer {
  height: 5px;
  cursor: row-resize;
  background: #ccc;
}

.map-panel {
  flex: 1;
  position: relative;
}

.map {
  width: 100%;
  height: 100%;
}

.navigation-info-panel {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 300px;
  background: white;
  border-left: 1px solid #ddd;
  overflow-y: auto;
  box-shadow: -2px 0 5px rgba(0,0,0,0.1);
}
.top-panel-content {
  display: flex;
  height: 100%;
}

.buttons-area {
  flex-shrink: 0;                /* 防止被挤小 */
  padding: 8px;
  background: #f9f9f9;
  border-right: 1px solid #ddd;
  display: flex;
  align-items: flex-start;
}

.left-column, .right-column {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.status-btn {
  padding: 6px 12px;
}

.status-btn.active {
  background: #007bff;
  color: white;
  border-radius: 4px;
}
.left-column button {
  padding: 6px 12px;
  font-size: 14px;
  border: none;
  border-radius: 6px;
  background: #4caf50; /* 默认绿色 */
  color: white;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  transition: background 0.2s, transform 0.1s;
}

.left-column button:nth-child(2) {
  background: #f0ad4e; /* 第二个按钮：黄色 */
}

.left-column button:nth-child(3) {
  background: #d9534f; /* 第三个按钮：红色 */
}

.left-column button:nth-child(4) {
  background: #2196f3; /* 第四个按钮：蓝色 */
}

.left-column button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.left-column button:active {
  transform: translateY(0);
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.dispatch-list-scrollable {
  flex: 1;                    /* 占满剩余空间 */
  overflow-y: auto;           /* 垂直滚动 */
  padding: 8px;
}
.btn-adopt {
  background-color: #5cb85c;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
}

.btn-reject{
  background-color: #d9534f;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 5px;
}
.btn-cancel {
  background-color: #f0ad4e;
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  margin-left: 5px;
}

.btn-nav {
  background-color: #2196f3;   /* 蓝色背景 */
  color: white;
  border: none;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.btn-nav:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}
.update-button {
  margin-top: 6px;
  width: fit-content;
  padding: 5px 12px;
  background-color: #409eff;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
}

.update-button:hover {
  background-color: #66b1ff;
}

</style>
