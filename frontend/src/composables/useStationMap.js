import { ref } from 'vue'
import request from '../api/axios'
import Map from 'ol/Map'
import View from 'ol/View'
import TileLayer from 'ol/layer/Tile'
import OSM from 'ol/source/OSM'
import VectorLayer from 'ol/layer/Vector'
import VectorSource from 'ol/source/Vector'
import Feature from 'ol/Feature'
import Point from 'ol/geom/Point'
import { fromLonLat } from 'ol/proj'
import { getStationStyle } from '../utils/mapStyles'

export function useStationMap() {
  const stations = ref([])
  const stationStatusMap = ref({})
  const loading = ref(false)


  // 用 ref 包裹
  const mapInstance = ref(null)
  let vectorLayer = null

  /**
   * 初始化地图
   */
  function initializeMap(targetElement, onMapClick) {
    if (!targetElement) {
      console.error('地图容器未找到')
      return
    }
    mapInstance.value = new Map({
      target: targetElement,
      layers: [ new TileLayer({ source: new OSM() }) ],
      view: new View({
        center: fromLonLat([-74.0576, 40.7312]),
        zoom: 14,
        maxZoom: 20,
        minZoom: 3
      }),
      controls: []
    })
    vectorLayer = new VectorLayer({ source: new VectorSource() })
    mapInstance.value.addLayer(vectorLayer)

    if (typeof onMapClick === 'function') {
      mapInstance.value.on('singleclick', evt => {
        let clickedStation = null
        mapInstance.value.forEachFeatureAtPixel(evt.pixel, feature => {
          const station = feature.get('stationData')
          if (station) {
            clickedStation = station
            return true
          }
        })
        if (clickedStation) {
          onMapClick(clickedStation)
        }
      })
    }
    console.log('地图初始化完成')
  }

 /**
 * 搜索并定位站点
 * @param {string} searchValue - 搜索关键词
 * @param {Array} stations - 站点数组
 * @param {Object} mapInstance - 地图实例
 * @param {Function} showAlert - 弹窗提示函数（alert 或自定义）
 */
function handleSearch(searchValue, stations, mapInstance, showAlert = alert) {
  const keyword = (searchValue || '').toLowerCase().trim()

  if (!keyword) {
    showAlert('请输入搜索内容')
    return
  }

  if (!stations || stations.length === 0) {
    showAlert('站点数据未加载')
    return
  }

  if (!mapInstance) {
    showAlert('地图未初始化')
    return
  }

  // 过滤匹配站点
  const matchedStations = stations.filter(station => {
    const name = station.station_name || ''
    const id = station.station_id || ''
    return name.toLowerCase().includes(keyword) || id.toLowerCase().includes(keyword)
  })

  if (matchedStations.length === 0) {
    showAlert('未找到相关站点')
    return
  }

  const station = matchedStations[0]

  const longitude = parseFloat(station.longitude)
  const latitude = parseFloat(station.latitude)

  if (isNaN(longitude) || isNaN(latitude)) {
    showAlert('站点坐标数据有误')
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
    showAlert('地图导航失败')
  }
}
  /**
   * 更新地图点位
   */
  function updateMapDisplay() {
    if (!mapInstance.value || !vectorLayer || !stations.value.length) {
      console.warn('地图未初始化或没有站点数据')
      return
    }
    vectorLayer.getSource().clear()
    const features = stations.value.map(station => {
      const status = stationStatusMap.value[station.station_id] || {}
      const bikeNum = status.stock ?? 0
      const feature = new Feature({
        geometry: new Point(fromLonLat([
          parseFloat(station.longitude), parseFloat(station.latitude)
        ]))
      })
      feature.setStyle(getStationStyle(bikeNum))
      feature.set('stationData', { ...station, bikeNum })
      return feature
    }).filter(Boolean)
    vectorLayer.getSource().addFeatures(features)
  }

  /**
   * 获取站点位置
   */
  async function fetchStationLocations() {
    try {
      loading.value = true
      const res = await request.get('/stations/locations')
      const data = res.data
      if (Array.isArray(data)) stations.value = data
      else if (data && Array.isArray(data.data)) stations.value = data.data
      else stations.value = []
    } catch (err) {
      console.error('获取站点位置失败:', err)
      stations.value = []
    } finally {
      loading.value = false
    }
  }

  /**
   * 获取站点状态
   */
  async function fetchAllStationsStatus(predictTime) {
    try {
      loading.value = true
      stationStatusMap.value = {}
      const res = await request.get('/predict/stations/all', {
        params: { predict_time: predictTime }
      })
      if (res.data?.stations_status?.length) {
        const map = {}
        res.data.stations_status.forEach(item => {
          map[item.station_id] = {
            stock: item.stock || 0,
            inflow: item.inflow || 0,
            outflow: item.outflow || 0
          }
        })
        stationStatusMap.value = map
      }
    } catch (err) {
      console.error('获取站点状态失败22:', err)
      stationStatusMap.value = {}
    } finally {
      loading.value = false
    }
  }

  return {
    stations,
    stationStatusMap,
    loading,
    mapInstance,
    initializeMap,
    updateMapDisplay,
    fetchStationLocations,
    fetchAllStationsStatus,
    handleSearch
  }
}
