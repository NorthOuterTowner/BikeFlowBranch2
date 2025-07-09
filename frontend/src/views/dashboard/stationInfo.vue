<script setup>
import { ref, watch, computed } from 'vue'
import request from '../../api/axios'

// Props
const props = defineProps({
  show: {
    type: Boolean,
    default: false
  },
  station: {
    type: Object,
    default: null
  },
  date: {
    type: String,
    default: ''
  },
  hour: {
    type: String,
    default: '00'
  }
})

// Emits
const emit = defineEmits(['update:show'])

// 状态
const loading = ref(false)
const actualBikeNum = ref(0)
const predictData = ref(null)
const error = ref('')

// 计算属性
const formattedDateTime = computed(() => {
  if (!props.date || !props.hour) return ''
  return `${props.date}T${props.hour}:00:00Z`
})

// 获取实际单车数量
async function fetchActualBikeNum() {
  if (!props.station?.station_id || !props.date || !props.hour) return
  
  try {
    const response = await request.get('/stations/bikeNum', {
      params: {
        station_id: props.station.station_id,
        date: props.date,
        hour: props.hour
      }
    })
    
    if (response.data && typeof response.data.bikeNum === 'number') {
      actualBikeNum.value = response.data.bikeNum
    } else if (typeof response.data === 'number') {
      actualBikeNum.value = response.data
    } else {
      actualBikeNum.value = 0
    }
  } catch (err) {
    console.error('获取实际单车数量失败:', err)
    actualBikeNum.value = 0
  }
}

// 获取预测数据
async function fetchPredictData() {
  if (!props.station?.station_id || !formattedDateTime.value) return
  
  try {
    const response = await request.get('/predict/station', {
      params: {
        station_id: props.station.station_id,
        predict_time: formattedDateTime.value
      }
    })
    
    if (response.data && response.data.status) {
      predictData.value = response.data
    } else {
      predictData.value = null
    }
  } catch (err) {
    console.error('获取预测数据失败:', err)
    predictData.value = null
  }
}

// 加载所有数据
async function loadStationData() {
  if (!props.station) return
  
  loading.value = true
  error.value = ''
  
  try {
    await Promise.all([
      fetchActualBikeNum(),
      fetchPredictData()
    ])
  } catch (err) {
    console.error('加载站点数据失败:', err)
    error.value = '数据加载失败'
  } finally {
    loading.value = false
  }
}

// 关闭弹窗
function closeDialog() {
  emit('update:show', false)
}

// 监听弹窗显示状态变化
watch(() => props.show, (newVal) => {
  if (newVal && props.station) {
    loadStationData()
  }
})

// 监听时间变化
watch([() => props.date, () => props.hour], () => {
  if (props.show && props.station) {
    loadStationData()
  }
})
</script>

<template>
  <div v-if="show" class="dialog-overlay" @click="closeDialog">
    <div class="dialog-content" @click.stop>
      <!-- 头部 -->
      <div class="dialog-header">
        <h3 class="dialog-title">站点详细信息</h3>
        <button class="close-button" @click="closeDialog">×</button>
      </div>
      
      <!-- 内容区域 -->
      <div class="dialog-body">
        <div v-if="loading" class="loading-section">
          <div class="loading-spinner"></div>
          <span>加载中...</span>
        </div>
        
        <div v-else-if="error" class="error-section">
          <span class="error-message">{{ error }}</span>
          <button class="retry-button" @click="loadStationData">重试</button>
        </div>
        
        <div v-else-if="station" class="station-info">
          <!-- 基本信息 -->
          <div class="info-section">
            <h4 class="section-title">基本信息</h4>
            <div class="info-grid">
              <div class="info-item">
                <label>站点编号:</label>
                <span>{{ station.station_id }}</span>
              </div>
              <div class="info-item">
                <label>站点名称:</label>
                <span>{{ station.station_name || '未知' }}</span>
              </div>
              <div class="info-item">
                <label>纬度:</label>
                <span>{{ station.latitude?.toFixed(4) }}</span>
              </div>
              <div class="info-item">
                <label>经度:</label>
                <span>{{ station.longitude?.toFixed(4) }}</span>
              </div>
            </div>
          </div>
          
          <!-- 时间信息 -->
          <div class="info-section">
            <h4 class="section-title">查询时间</h4>
            <div class="info-grid">
              <div class="info-item">
                <label>日期:</label>
                <span>{{ date }}</span>
              </div>
              <div class="info-item">
                <label>时间:</label>
                <span>{{ hour }}:00</span>
              </div>
            </div>
          </div>
          
          <!-- 实际数据 -->
          <div class="info-section">
            <h4 class="section-title">实际数据</h4>
            <div class="data-card actual-data">
              <div class="data-item">
                <span class="data-label">实际单车数量</span>
                <span class="data-value actual">{{ actualBikeNum }}</span>
              </div>
            </div>
          </div>
          
          <!-- 预测数据 -->
          <div class="info-section">
            <h4 class="section-title">预测数据</h4>
            <div v-if="predictData" class="data-card predict-data">
              <div class="data-item">
                <span class="data-label">预测单车数量</span>
                <span class="data-value predict">{{ predictData.status.stock }}</span>
              </div>
              <div class="data-item">
                <span class="data-label">预测入车流</span>
                <span class="data-value inflow">{{ predictData.status.inflow }}</span>
              </div>
              <div class="data-item">
                <span class="data-label">预测出车流</span>
                <span class="data-value outflow">{{ predictData.status.outflow }}</span>
              </div>
            </div>
            <div v-else class="no-data">
              <span>暂无预测数据</span>
            </div>
          </div>
          
          <!-- 对比分析 -->
          <div v-if="predictData" class="info-section">
            <h4 class="section-title">对比分析</h4>
            <div class="comparison-card">
              <div class="comparison-item">
                <span class="comparison-label">实际 vs 预测</span>
                <div class="comparison-values">
                  <span class="actual-value">{{ actualBikeNum }}</span>
                  <span class="vs">vs</span>
                  <span class="predict-value">{{ predictData.status.stock }}</span>
                </div>
              </div>
              <div class="comparison-item">
                <span class="comparison-label">差异</span>
                <span class="difference-value" :class="{ 
                  'positive': (actualBikeNum - predictData.status.stock) > 0,
                  'negative': (actualBikeNum - predictData.status.stock) < 0,
                  'zero': (actualBikeNum - predictData.status.stock) === 0
                }">
                  {{ actualBikeNum - predictData.status.stock > 0 ? '+' : '' }}{{ actualBikeNum - predictData.status.stock }}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

  .dialog-content {
  background: white;
  border-radius: 8px;
  width: 600px;
  max-height: 90%;
  overflow-y: auto;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  padding: 20px;
  position: relative;
}

.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #ddd;
  padding-bottom: 10px;
  margin-bottom: 15px;
}

.dialog-title {
  font-size: 18px;
  font-weight: bold;
}

.close-button {
  background: none;
  border: none;
  font-size: 24px;
  cursor: pointer;
}

.dialog-body {
  font-size: 14px;
  color: #333;
}

.info-section {
  margin-bottom: 20px;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
}

.info-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.info-item {
  width: 48%;
  display: flex;
  justify-content: space-between;
}

.data-card {
  background: #f9f9f9;
  padding: 12px;
  border-radius: 6px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.data-item {
  display: flex;
  justify-content: space-between;
}

.data-label {
  font-weight: 500;
}

.data-value {
  font-weight: bold;
}

.comparison-card {
  background: #f0f4ff;
  padding: 12px;
  border-radius: 6px;
}

.comparison-item {
  margin-bottom: 10px;
}

.comparison-label {
  font-weight: 600;
}

.comparison-values {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 5px;
}

.vs {
  color: #999;
}

.difference-value {
  font-weight: bold;
  font-size: 16px;
}

.difference-value.positive {
  color: green;
}

.difference-value.negative {
  color: red;
}

.difference-value.zero {
  color: #333;
}

.loading-section,
.error-section {
  text-align: center;
  margin: 30px 0;
}

.retry-button {
  margin-top: 10px;
  padding: 5px 10px;
  background: #409eff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
</style>
