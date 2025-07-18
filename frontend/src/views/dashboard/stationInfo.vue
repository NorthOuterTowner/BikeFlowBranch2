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
  <Transition name="dialog-fade">
    <div v-if="show" class="dialog-overlay" @click="closeDialog">
      <Transition name="dialog-slide">
        <div class="dialog-content" @click.stop>
          <!-- 头部 -->
          <div class="dialog-header">
            <div class="header-info">
              <div class="station-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                  <path d="M2 17l10 5 10-5"/>
                  <path d="M2 12l10 5 10-5"/>
                </svg>
              </div>
              <div>
                <h3 class="dialog-title">站点详细信息</h3>
                <p class="station-subtitle">{{ station?.station_name || '未知站点' }}</p>
              </div>
            </div>
            <button class="close-button" @click="closeDialog">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="18" y1="6" x2="6" y2="18"/>
                <line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
          </div>
          
          <!-- 内容区域 -->
          <div class="dialog-body">
            <div v-if="loading" class="loading-section">
              <div class="loading-spinner">
                <div class="spinner"></div>
              </div>
              <span class="loading-text">加载中...</span>
            </div>
            
            <div v-else-if="error" class="error-section">
              <div class="error-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="12" y1="8" x2="12" y2="12"/>
                  <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
              </div>
              <span class="error-message">{{ error }}</span>
              <button class="retry-button" @click="loadStationData">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M23 4v6h-6"/>
                  <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                </svg>
                重试
              </button>
            </div>
            
            <div v-else-if="station" class="station-info">
              <!-- 基本信息 -->
              <div class="info-section">
                <div class="section-header">
                  <div class="section-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M3 3h18v18H3zM9 9h6v6H9z"/>
                    </svg>
                  </div>
                  <h4 class="section-title">基本信息</h4>
                </div>
                <div class="info-grid">
                  <div class="info-item">
                    <div class="info-label">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2z"/>
                        <path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                      </svg>
                      站点编号
                    </div>
                    <span class="info-value">{{ station.station_id }}</span>
                  </div>
                  <div class="info-item">
                    <div class="info-label">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
                        <circle cx="12" cy="10" r="3"/>
                      </svg>
                      站点名称
                    </div>
                    <span class="info-value">{{ station.station_name || '未知' }}</span>
                  </div>
                  <div class="info-item">
                    <div class="info-label">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      纬度
                    </div>
                    <span class="info-value">{{ station.latitude?.toFixed(4) }}</span>
                  </div>
                  <div class="info-item">
                    <div class="info-label">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      经度
                    </div>
                    <span class="info-value">{{ station.longitude?.toFixed(4) }}</span>
                  </div>
                </div>
              </div>
              
              <!-- 时间信息 -->
              <div class="info-section">
                <div class="section-header">
                  <div class="section-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <circle cx="12" cy="12" r="10"/>
                      <polyline points="12,6 12,12 16,14"/>
                    </svg>
                  </div>
                  <h4 class="section-title">查询时间</h4>
                </div>
                <div class="time-display">
                  <div class="time-item">
                    <div class="time-label">日期</div>
                    <div class="time-value">{{ date }}</div>
                  </div>
                  <div class="time-divider">|</div>
                  <div class="time-item">
                    <div class="time-label">时间</div>
                    <div class="time-value">{{ hour }}:00</div>
                  </div>
                </div>
              </div>
              
              <!-- 数据对比卡片 -->
              <div class="data-comparison">
                <div class="data-card actual-card">
                  <div class="card-header">
                    <div class="card-icon actual-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 11H5a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-7a2 2 0 0 0-2-2z"/>
                        <path d="M19 7h-4a2 2 0 0 0-2 2v11a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2z"/>
                      </svg>
                    </div>
                    <div class="card-title">实际数据</div>
                  </div>
                  <div class="card-value actual-value">{{ actualBikeNum }}</div>
                  <div class="card-label">实际单车数量</div>
                </div>
                
                <div v-if="predictData" class="data-card predict-card">
                  <div class="card-header">
                    <div class="card-icon predict-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                      </svg>
                    </div>
                    <div class="card-title">预测数据</div>
                  </div>
                  <div class="card-value predict-value">{{ predictData.status.stock }}</div>
                  <div class="card-label">预测单车数量</div>
                </div>
              </div>
              
              <!-- 预测流量数据 -->
              <div v-if="predictData" class="flow-section">
                <div class="section-header">
                  <div class="section-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/>
                    </svg>
                  </div>
                  <h4 class="section-title">流量预测</h4>
                </div>
                <div class="flow-grid">
                  <div class="flow-item inflow">
                    <div class="flow-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 19V5m-7 7l7-7 7 7"/>
                      </svg>
                    </div>
                    <div class="flow-content">
                      <div class="flow-value">{{ predictData.status.inflow }}</div>
                      <div class="flow-label">预测入车流</div>
                    </div>
                  </div>
                  <div class="flow-item outflow">
                    <div class="flow-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 5v14m7-7-7 7-7-7"/>
                      </svg>
                    </div>
                    <div class="flow-content">
                      <div class="flow-value">{{ predictData.status.outflow }}</div>
                      <div class="flow-label">预测出车流</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- 对比分析 -->
              <div v-if="predictData" class="analysis-section">
                <div class="section-header">
                  <div class="section-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M18 20V10"/>
                      <path d="M12 20V4"/>
                      <path d="M6 20v-6"/>
                    </svg>
                  </div>
                  <h4 class="section-title">对比分析</h4>
                </div>
                <div class="analysis-card">
                  <div class="comparison-display">
                    <div class="comparison-item">
                      <span class="comparison-label">实际数量</span>
                      <span class="actual-number">{{ actualBikeNum }}</span>
                    </div>
                    <div class="vs-divider">VS</div>
                    <div class="comparison-item">
                      <span class="comparison-label">预测数量</span>
                      <span class="predict-number">{{ predictData.status.stock }}</span>
                    </div>
                  </div>
                  <div class="difference-display">
                    <div class="difference-label">差异值</div>
                    <div class="difference-value" :class="{ 
                      'positive': (predictData.status.stock-actualBikeNum ) > 0,
                      'negative': (predictData.status.stock-actualBikeNum ) < 0,
                      'zero': (predictData.status.stock-actualBikeNum) === 0
                    }">
                      <span class="difference-sign">{{ predictData.status.stock-actualBikeNum > 0 ? '+' : predictData.status.stock-actualBikeNum < 0 ? '-' : '' }}</span>
                      <span class="difference-number">{{ Math.abs(predictData.status.stock-actualBikeNum) }}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- 暂无预测数据 -->
              <div v-if="!predictData" class="no-data-section">
                <div class="no-data-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
                    <path d="M12 17h.01"/>
                  </svg>
                </div>
                <span class="no-data-text">暂无预测数据</span>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </div>
  </Transition>
</template>

<style scoped>
/* 过渡动画 */
.dialog-fade-enter-active, .dialog-fade-leave-active {
  transition: opacity 0.3s ease;
}

.dialog-fade-enter-from, .dialog-fade-leave-to {
  opacity: 0;
}

.dialog-slide-enter-active, .dialog-slide-leave-active {
  transition: all 0.3s ease;
}

.dialog-slide-enter-from {
  transform: translateY(-30px) scale(0.95);
  opacity: 0;
}

.dialog-slide-leave-to {
  transform: translateY(30px) scale(0.95);
  opacity: 0;
}

/* 基础样式 */
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(8px);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  padding: 20px;
}

.dialog-content {
  background: linear-gradient(135deg, #ffffff 0%, #f8fafc 30%, #e2e8f0 70%, #cbd5e1 100%);
  border-radius: 20px;
  width: 100%;
  max-width: 700px;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

/* 头部样式 */
.dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 28px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.5);
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  color: white;
  border-radius: 20px 20px 0 0;
  
}

.header-info {
  display: flex;
  align-items: center;
  gap: 16px;
}

.station-icon {
  width: 48px;
  height: 48px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
}

.station-icon svg {
  width: 24px;
  height: 24px;
}

.dialog-title {
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 4px 0;
}

.station-subtitle {
  font-size: 14px;
  margin: 0;
  opacity: 0.9;
}

.close-button {
  width: 40px;
  height: 40px;
  background: rgba(255, 255, 255, 0.2);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  backdrop-filter: blur(10px);
}

.close-button:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: scale(1.05);
}

.close-button svg {
  width: 20px;
  height: 20px;
}

/* 内容区域 */
.dialog-body {
  padding: 28px;
  font-size: 14px;
  color: #334155;
}

/* 加载状态 */
.loading-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  gap: 20px;
}

.loading-spinner {
  position: relative;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #e2e8f0;
  border-top: 3px solid #091275;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.loading-text {
  font-size: 16px;
  color: #64748b;
  font-weight: 500;
}

/* 错误状态 */
.error-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  gap: 20px;
}

.error-icon {
  width: 48px;
  height: 48px;
  color: #ef4444;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(239, 68, 68, 0.1);
  border-radius: 12px;
}

.error-icon svg {
  width: 24px;
  height: 24px;
}

.error-message {
  font-size: 16px;
  color: #ef4444;
  font-weight: 500;
}

.retry-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.retry-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(9, 18, 117, 0.3);
}

.retry-button svg {
  width: 16px;
  height: 16px;
}

/* 信息区块 */
.info-section {
  margin-bottom: 32px;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.section-icon {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  color: white;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.section-icon svg {
  width: 16px;
  height: 16px;
}

.section-title {
  font-size: 18px;
  font-weight: 600;
  color: #1e293b;
  margin: 0;
}

/* 信息网格 */
.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 16px;
}

.info-item {
  background: white;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: all 0.2s ease;
}

.info-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.info-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
  color: #64748b;
}

.info-label svg {
  width: 16px;
  height: 16px;
}

.info-value {
  font-weight: 600;
  color: #1e293b;
}

/* 时间显示 */
.time-display {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 24px;
  background: white;
  padding: 24px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}

.time-item {
  text-align: center;
}

.time-label {
  font-size: 12px;
  color: #64748b;
  font-weight: 500;
  margin-bottom: 8px;
}

.time-value {
  font-size: 20px;
  font-weight: 700;
  color: #1e293b;
}

.time-divider {
  font-size: 20px;
  color: #cbd5e1;
  font-weight: 300;
}

/* 数据对比卡片 */
.data-comparison {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.data-card {
  background: white;
  padding: 28px;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  text-align: center;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.data-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
}

.data-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 20px;
}

.card-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.actual-icon {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.predict-icon {
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
}

.card-icon svg {
  width: 20px;
  height: 20px;
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: #1e293b;
}

.card-value {
  font-size: 36px;
  font-weight: 800;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.card-label {
  font-size: 14px;
  color: #64748b;
  font-weight: 500;
}

/* 流量数据 */
.flow-section {
  margin-bottom: 32px;
}

.flow-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.flow-item {
  background: white;
  padding: 24px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.2s ease;
}

.flow-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.flow-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.inflow .flow-icon {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.outflow .flow-icon {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.flow-icon svg {
  width: 24px;
  height: 24px;
}

.flow-content {
  flex: 1;
}

.flow-value {
  font-size: 24px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 4px;
}

.flow-label {
  font-size: 14px;
  color: #64748b;
  font-weight: 500;
}

/* 分析区块 */
.analysis-section {
  margin-bottom: 32px;
}

.analysis-card {
  background: white;
  padding: 32px;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  position: relative;
}

.analysis-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  border-radius: 16px 16px 0 0;
}

.comparison-display {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding: 20px;
  background: #f8fafc;
  border-radius: 12px;
}

.comparison-item {
  text-align: center;
  flex: 1;
}

.comparison-label {
  font-size: 12px;
  color: #64748b;
  font-weight: 500;
  margin-bottom: 8px;
  display: block;
}

.actual-number {
  font-size: 28px;
  font-weight: 700;
  color: #10b981;
}

.predict-number {
  font-size: 28px;
  font-weight: 700;
  color: #3b82f6;
}

.vs-divider {
  font-size: 16px;
  font-weight: 600;
  color: #94a3b8;
  background: white;
  padding: 8px 16px;
  border-radius: 8px;
  border: 2px solid #e2e8f0;
}

.difference-display {
  text-align: center;
  padding: 20px;
  background: #f1f5f9;
  border-radius: 12px;
}

.difference-label {
  font-size: 14px;
  color: #64748b;
  font-weight: 500;
  margin-bottom: 12px;
}

.difference-value {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  font-size: 32px;
  font-weight: 800;
}

.difference-value.positive {
  color: #10b981;
}

.difference-value.negative {
  color: #ef4444;
}

.difference-value.zero {
  color: #64748b;
}

.difference-sign {
  font-size: 24px;
}

.difference-number {
  font-size: 32px;
}

/* 无数据状态 */
.no-data-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  gap: 20px;
  background: white;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
}

.no-data-icon {
  width: 64px;
  height: 64px;
  color: #94a3b8;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f1f5f9;
  border-radius: 16px;
}

.no-data-icon svg {
  width: 32px;
  height: 32px;
}

.no-data-text {
  font-size: 16px;
  color: #64748b;
  font-weight: 500;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .dialog-overlay {
    padding: 16px;
  }
  
  .dialog-content {
    max-width: 100%;
    max-height: 95vh;
    border-radius: 16px;
  }
  
  .dialog-header {
    padding: 20px;
    border-radius: 16px 16px 0 0;
  }
  
  .dialog-body {
    padding: 20px;
  }
  
  .header-info {
    gap: 12px;
  }
  
  .station-icon {
    width: 40px;
    height: 40px;
  }
  
  .dialog-title {
    font-size: 18px;
  }
  
  .info-grid {
    grid-template-columns: 1fr;
  }
  
  .data-comparison {
    grid-template-columns: 1fr;
  }
  
  .flow-grid {
    grid-template-columns: 1fr;
  }
  
  .time-display {
    flex-direction: column;
    gap: 16px;
  }
  
  .time-divider {
    display: none;
  }
  
  .comparison-display {
    flex-direction: column;
    gap: 16px;
  }
  
  .vs-divider {
    order: 2;
  }
}

@media (max-width: 480px) {
  .dialog-header {
    padding: 16px;
  }
  
  .dialog-body {
    padding: 16px;
  }
  
  .header-info {
    flex-direction: column;
    text-align: center;
    gap: 8px;
  }
  
  .station-icon {
    width: 36px;
    height: 36px;
  }
  
  .dialog-title {
    font-size: 16px;
  }
  
  .station-subtitle {
    font-size: 13px;
  }
  
  .section-title {
    font-size: 16px;
  }
  
  .card-value {
    font-size: 28px;
  }
  
  .flow-value {
    font-size: 20px;
  }
  
  .actual-number, .predict-number {
    font-size: 24px;
  }
  
  .difference-number {
    font-size: 28px;
  }
}

/* 滚动条样式 */
.dialog-content::-webkit-scrollbar {
  width: 6px;
}

.dialog-content::-webkit-scrollbar-track {
  background: #f1f5f9;
  border-radius: 3px;
}

.dialog-content::-webkit-scrollbar-thumb {
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  border-radius: 3px;
}

.dialog-content::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(135deg, #4953c2 0%, #0e177e 100%);
}

/* 微动画效果 */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.station-info > .info-section {
  animation: fadeInUp 0.5s ease forwards;
}

.station-info > .info-section:nth-child(1) {
  animation-delay: 0.1s;
}

.station-info > .info-section:nth-child(2) {
  animation-delay: 0.2s;
}

.station-info > .data-comparison {
  animation: fadeInUp 0.5s ease forwards;
  animation-delay: 0.3s;
}

.station-info > .flow-section {
  animation: fadeInUp 0.5s ease forwards;
  animation-delay: 0.4s;
}

.station-info > .analysis-section {
  animation: fadeInUp 0.5s ease forwards;
  animation-delay: 0.5s;
}

/* 悬浮效果增强 */
.data-card:hover .card-icon {
  transform: scale(1.1);
  transition: transform 0.2s ease;
}

.flow-item:hover .flow-icon {
  transform: scale(1.05);
  transition: transform 0.2s ease;
}

.info-item:hover .info-label svg {
  transform: scale(1.1);
  transition: transform 0.2s ease;
}

/* 渐变背景动画 */
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

.dialog-header {
  background: linear-gradient(135deg, #4953c2 0%, #29148f 50%, #091275 100%);
  background-size: 200% 200%;
  animation: gradientShift 6s ease infinite;
}
</style>