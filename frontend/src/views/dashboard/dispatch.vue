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
const currentHour = localStorage.getItem('selectedHour') // 获取当前小时字符串

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

//编辑调度
const editingDispatch = ref(null) // 当前正在编辑的调度方案
const editQuantity = ref(0) // 编辑中的数量
const showEditDialog = ref(false) // 是否显示编辑对话框

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
/**
 * 编辑调度数量接口调用
 */
async function editDispatchQuantity(id, bikes) {
  try {
    const response = await request.post('/dispatch/edit', {
      id: id,
      bikes: bikes
    })
    
    if (response.data?.code === 200) {
      return { success: true, message: response.data.msg || '修改成功' }
    } else {
      return { success: false, message: response.data?.msg || '修改失败' }
    }
  } catch (error) {
    console.error('编辑调度数量失败:', error)
    return { success: false, message: '网络错误，请稍后重试' }
  }
}

/**
 * 打开编辑对话框
 */
function openEditDialog(item) {
  editingDispatch.value = item
  editQuantity.value = item.bikes_to_move || 0
  showEditDialog.value = true
}

/**
 * 关闭编辑对话框
 */
function closeEditDialog() {
  editingDispatch.value = null
  editQuantity.value = 0
  showEditDialog.value = false
}

/**
 * 确认保存编辑
 */
async function saveEditQuantity() {
  if (!editingDispatch.value || editQuantity.value <= 0) {
    alert('请输入有效的调度数量')
    return
  }
  
  try {
    const result = await editDispatchQuantity(
      editingDispatch.value.schedule_id, 
      editQuantity.value
    )
    
    if (result.success) {
      // 更新本地数据
      const index = allDispatchList.value.findIndex(
        item => item.schedule_id === editingDispatch.value.schedule_id
      )
      if (index !== -1) {
        allDispatchList.value[index].bikes_to_move = editQuantity.value
      }
      
      // 如果当前选中的是被编辑的方案，重新绘制地图
      if (selectedPlan.value && selectedPlan.value.schedule_id === editingDispatch.value.schedule_id) {
        selectedPlan.value.bikes_to_move = editQuantity.value
        drawDispatchPlanOnMap(selectedPlan.value)
      }
      
      closeEditDialog()
      
      // 显示成功提示
      if (typeof ElMessage !== 'undefined') {
        ElMessage.success(result.message)
      } else {
        alert(result.message)
      }
    } else {
      alert(result.message)
    }
  } catch (error) {
    console.error('保存编辑失败:', error)
    alert('保存失败，请稍后重试')
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
    const stations = await getStationAssign({ date: fixedDate.value, hour:currentHour});
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
      src: '/icons/arrow.png',
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
 * 构建查询时间字符串
 * @param {string} date - 日期 (YYYY-MM-DD)
 * @param {string} hour - 小时 (HH:mm)
 * @returns {string} ISO 8601格式的时间字符串
 */
function buildQueryTime(date, hour) {
  try {
    let hourStr = hour.toString()
    
    // 如果hour只是数字，转换为HH:00格式
    if (!/\d{1,2}:\d{2}/.test(hourStr)) {
      const hourNum = parseInt(hourStr)
      hourStr = hourNum.toString().padStart(2, '0') + ':00'
    }
    
    // 构建ISO 8601格式的时间字符串
    const isoString = `${date}T${hourStr}:00Z`
    
    console.log('构建查询时间:', { date, hour, hourStr, isoString })
    return isoString
  } catch (error) {
    console.error('构建查询时间失败:', error)
    // 返回当前时间作为fallback
    return new Date().toISOString()
  }
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
    const query_time= buildQueryTime(fixedDate.value, currentHour)
    const res = await getDispatch(query_time)
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
      dispatchDate: fixedDate.value,
      dispatchHour: currentHour,
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
      dispatchDate: fixedDate.value,
      dispatchHour: currentHour,
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
      <!-- table 部分 -->
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
            <td class="col-number">
              <div class="quantity-control">
                <div class="quantity-display">
                  {{ item.bikes_to_move ?? '-' }}
                </div>
                <div class="quantity-buttons" v-if="item.statusInt === '待执行'">
                  <button 
                    class="qty-btn qty-edit" 
                    @click.stop="openEditDialog(item)"
                  >
                    编辑
                  </button>
                </div>
              </div>
            </td>
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

      <!-- 编辑数量对话框 -->
      <div v-if="showEditDialog" class="edit-dialog-overlay" @click="closeEditDialog">
        <div class="edit-dialog" @click.stop>
          <div class="edit-dialog-header">
            <h3>编辑调度数量</h3>
            <button class="dialog-close-btn" @click="closeEditDialog">×</button>
          </div>
          <div class="edit-dialog-content">
            <div class="edit-info">
              <div class="edit-route">
                <span class="route-label">调度路线：</span>
                <span class="route-text">
                  {{ editingDispatch?.start_station.name }} → {{ editingDispatch?.end_station.name }}
                </span>
              </div>
              <div class="edit-current">
                <span class="current-label">当前数量：</span>
                <span class="current-value">{{ editingDispatch?.bikes_to_move || 0 }} 辆</span>
              </div>
            </div>
            <div class="edit-input-group">
              <label for="quantity-input">新数量：</label>
              <input 
                id="quantity-input"
                type="number" 
                v-model.number="editQuantity" 
                min="1"
                max="999"
                class="quantity-input"
                @keyup.enter="saveEditQuantity"
              />
              <span class="input-unit">辆</span>
            </div>
          </div>
     <div class="edit-dialog-footer">
      <button class="dialog-btn dialog-btn-cancel" @click="closeEditDialog">取消</button>
      <button class="dialog-btn dialog-btn-save" @click="saveEditQuantity">保存</button>
    </div>
  </div>
</div>
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

/* 数量控制区域样式 */
.quantity-control {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.quantity-display {
  font-weight: 600;
  font-size: 14px;
  color: #333;
  min-height: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.quantity-buttons {
  display: flex;
  gap: 2px;
  flex-wrap: wrap;
  justify-content: center;
}

.qty-btn {
  padding: 2px 6px;
  border: none;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
  min-width: 20px;
  height: 20px;
  line-height: 1;
  transition: all 0.2s ease;
}

.qty-edit {
  background-color: #2196f3;
  color: white;
  font-size: 10px;
  padding: 2px 4px;
}

.qty-edit:hover {
  background-color: #1976d2;
}

/* 编辑对话框样式 - 美化版 */
.edit-dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.4));
  backdrop-filter: blur(8px);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
  animation: fadeIn 0.3s ease-out;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.edit-dialog {
  background: linear-gradient(145deg, #ffffff, #f8f9fa);
  border-radius: 16px;
  width: 420px;
  max-width: 90vw;
  box-shadow: 
    0 20px 40px rgba(0, 0, 0, 0.1),
    0 8px 16px rgba(0, 0, 0, 0.08),
    0 2px 4px rgba(0, 0, 0, 0.06);
  overflow: hidden;
  transform: translateY(-20px);
  animation: slideIn 0.3s ease-out forwards;
  border: 1px solid rgba(255, 255, 255, 0.8);
}

@keyframes slideIn {
  from {
    transform: translateY(-20px);
    opacity: 0;
  }
  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.edit-dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  background-size: 200% 200%;
  animation: gradientShift 4s ease infinite;
  border-bottom: none;
  position: relative;
}

@keyframes gradientShift {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

.edit-dialog-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), transparent);
  pointer-events: none;
}

.edit-dialog-header h3 {
  margin: 0;
  font-size: 20px;
  font-weight: 700;
  color: #ffffff;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  letter-spacing: 0.5px;
}

.dialog-close-btn {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  font-size: 20px;
  color: #ffffff;
  cursor: pointer;
  padding: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;
}

.dialog-close-btn::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  transition: all 0.3s ease;
}

.dialog-close-btn:hover {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  transform: scale(1.1);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.dialog-close-btn:hover::before {
  width: 100%;
  height: 100%;
}

.dialog-close-btn:active {
  transform: scale(0.95);
}

.edit-dialog-content {
  padding: 24px;
  background: #ffffff;
  position: relative;
}

.edit-dialog-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
}

/* 响应式设计 */
@media (max-width: 480px) {
  .edit-dialog {
    width: 95vw;
    margin: 20px;
  }
  
  .edit-dialog-header {
    padding: 16px 20px;
  }
  
  .edit-dialog-header h3 {
    font-size: 18px;
  }
  
  .dialog-close-btn {
    width: 32px;
    height: 32px;
    font-size: 18px;
  }
  
  .edit-dialog-content {
    padding: 20px;
  }
}

/* 深色主题支持 */
@media (prefers-color-scheme: dark) {
  .edit-dialog {
    background: linear-gradient(145deg, #2d3748, #1a202c);
    border-color: rgba(255, 255, 255, 0.1);
  }
  
  .edit-dialog-content {
    background: #2d3748;
    color: #e2e8f0;
  }
  
  .edit-dialog-content::before {
    background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.4), transparent);
  }
}

/* 高级动画效果 */
.edit-dialog:hover {
  box-shadow: 
    0 25px 50px rgba(0, 0, 0, 0.15),
    0 12px 20px rgba(0, 0, 0, 0.1),
    0 4px 8px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
  transition: all 0.3s ease;
}

/* 无障碍支持 */
@media (prefers-reduced-motion: reduce) {
  .edit-dialog-overlay,
  .edit-dialog,
  .dialog-close-btn,
  .edit-dialog-header {
    animation: none;
    transition: none;
  }
}

/* 焦点样式 */
.dialog-close-btn:focus {
  outline: none;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.5);
}

.dialog-close-btn:focus:not(:focus-visible) {
  box-shadow: none;
}


.edit-dialog-content {
  padding: 20px;
}

.edit-info {
  margin-bottom: 20px;
  padding: 12px;
  background-color: #f8f9fa;
  border-radius: 6px;
}

.edit-route, .edit-current {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.edit-route:last-child, .edit-current:last-child {
  margin-bottom: 0;
}

.route-label, .current-label {
  font-weight: 600;
  color: #495057;
  min-width: 80px;
}

.route-text {
  color: #5a626c;
  font-weight: 500;
}

.current-value {
  color: #17116b;
  font-weight: 600;
}

.edit-input-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.edit-input-group label {
  font-weight: 600;
  color: #495057;
  min-width: 60px;
}

.quantity-input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 14px;
  text-align: center;
}

.quantity-input:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
}

.input-unit {
  color: #6c757d;
  font-size: 14px;
}

.edit-dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 20px;
  background-color: #f8f9fa;
  border-top: 1px solid #e9ecef;
}

.dialog-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.dialog-btn-cancel {
  background-color: #6c757d;
  color: white;
}

.dialog-btn-cancel:hover {
  background-color: #5a6268;
}

.dialog-btn-save {
  background-color: #2f1fac;
  color: white;
}

.dialog-btn-save:hover {
  background-color: #150c61;
}

/* 调整数量列宽度 */
.col-number {
  width: 120px;
  text-align: center;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .quantity-buttons {
    flex-direction: column;
    gap: 1px;
  }
  
  .qty-btn {
    width: 100%;
    min-width: 40px;
  }
  
  .edit-dialog {
    width: 90vw;
    margin: 20px;
  }
  
  .edit-input-group {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  
  .edit-input-group label {
    min-width: auto;
  }
}

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

.map :deep(.ol-zoom-custom) {
  position: absolute;
  bottom: 20px;
  right: 20px;
  z-index: 1000;
}

.map :deep(.ol-zoom-custom button) {
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

.map :deep(.ol-zoom-custom button:hover) {
  background-color: #f0f0f0;
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
