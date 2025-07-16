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
const inboundData = ref(null)
const outboundData = ref(null)
const error = ref('')

// 计算属性
const formattedDateTime = computed(() => {
  if (!props.date || !props.hour) return ''
  const hourOnly = props.hour.split(':')[0]  // 去掉 ":00" 部分
  return `${props.date}T${hourOnly.padStart(2, '0')}:00:00Z`
})

// 计算调入任务（站点作为终点）
const inboundSchedules = computed(() => {
  return inboundData.value?.schedules || []
})

// 计算调出任务（站点作为起点）
const outboundSchedules = computed(() => {
  return outboundData.value?.schedules || []
})

// 工具函数
function getStatusColor(status) {
  const statusColors = {
    // 中文状态值映射
    '待执行': '#f59e0b',     // 待执行 - 橙色
    '正在执行': '#3b82f6',     // 执行中 - 蓝色
    '已完成': '#10b981',     // 已完成 - 绿色
    '已取消': '#ef4444',     // 已取消 - 红色
    // 英文状态值映射（兼容）
    'pending': '#f59e0b',
    'in_progress': '#3b82f6',
    'completed': '#10b981',
    'cancelled': '#ef4444',
  }
  return statusColors[status] || '#6b7280' // 默认灰色
}

function getStatusText(status) {
  // 如果已经是中文状态，直接返回
  const chineseStatuses = ['待执行', '正在执行', '已完成', '已取消']
  if (chineseStatuses.includes(status)) {
    return status
  }
  
  // 英文状态转中文
  const statusTexts = {
    'pending': '待执行',
    'in_progress': '正在执行',
    'completed': '已完成',
    'cancelled': '已取消',
  }
  return statusTexts[status] || status
}

function formatDateTime(dateTime) {
  if (!dateTime) return ''
  return new Date(dateTime).toLocaleString('zh-CN')
}

// 获取调入数据（站点作为终点）
async function fetchInboundData() {
  if (!props.station?.station_id || !formattedDateTime.value) return
  
  try {
    const response = await request.get('/dispatch/by-station', {
      params: {
        station_id: props.station.station_id,
        query_time: formattedDateTime.value,
        role: 'end'  // 站点作为终点，即调入任务
      }
    })
    
    inboundData.value = response.data
  } catch (err) {
    console.error('获取调入数据失败:', err)
    inboundData.value = null
  }
}

// 获取调出数据（站点作为起点）
async function fetchOutboundData() {
  if (!props.station?.station_id || !formattedDateTime.value) return
  
  try {
    const response = await request.get('/dispatch/by-station', {
      params: {
        station_id: props.station.station_id,
        query_time: formattedDateTime.value,
        role: 'start'  // 站点作为起点，即调出任务
      }
    })
    
    outboundData.value = response.data
  } catch (err) {
    console.error('获取调出数据失败:', err)
    outboundData.value = null
  }
}

// 加载数据
async function loadStationData() {
  if (!props.station) return
  
  loading.value = true
  error.value = ''
  
  try {
    // 并行请求调入和调出数据
    await Promise.all([
      fetchInboundData(),
      fetchOutboundData()
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
                  <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                  <polyline points="9,22 9,12 15,12 15,22"/>
                </svg>
              </div>
              <div>
                <h3 class="dialog-title">调度信息</h3>
                <p class="station-subtitle">{{ station?.station_name || '未知站点' }}</p>
                <p class="time-subtitle">{{ date }} {{ hour }}:00</p>
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
            
            <div v-else class="dispatch-content">
              <!-- 调度统计概览 -->
              <div class="dispatch-overview">
                <div class="overview-item">
                  <div class="overview-icon total-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
                      <circle cx="9" cy="7" r="4"/>
                      <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
                      <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                    </svg>
                  </div>
                  <div class="overview-content">
                    <div class="overview-value">{{ (inboundSchedules.length + outboundSchedules.length) }}</div>
                    <div class="overview-label">总调度任务</div>
                  </div>
                </div>
                
                <div class="overview-item">
                  <div class="overview-icon in-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>
                      <polyline points="8,7 3,12 8,17"/>
                      <line x1="3" y1="12" x2="15" y2="12"/>
                    </svg>
                  </div>
                  <div class="overview-content">
                    <div class="overview-value">{{ inboundSchedules.length }}</div>
                    <div class="overview-label">调入任务</div>
                  </div>
                </div>
                
                <div class="overview-item">
                  <div class="overview-icon out-icon">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                      <polyline points="16,17 21,12 16,7"/>
                      <line x1="21" y1="12" x2="9" y2="12"/>
                    </svg>
                  </div>
                  <div class="overview-content">
                    <div class="overview-value">{{ outboundSchedules.length }}</div>
                    <div class="overview-label">调出任务</div>
                  </div>
                </div>
              </div>

              <!-- 调度表格区域 -->
              <div v-if="inboundSchedules.length > 0 || outboundSchedules.length > 0" class="dispatch-tables">
                <!-- 调入表格 -->
                <div v-if="inboundSchedules.length > 0" class="dispatch-table-section">
                  <div class="table-header inbound-header">
                    <div class="table-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/>
                        <polyline points="8,7 3,12 8,17"/>
                        <line x1="3" y1="12" x2="15" y2="12"/>
                      </svg>
                    </div>
                    <h5 class="table-title">调入任务 ({{ inboundSchedules.length }})</h5>
                  </div>
                  <div class="table-container">
                    <table class="dispatch-table">
                      <thead>
                        <tr>
                          <th>调度ID</th>
                          <th>来源站点</th>
                          <th>调度数量</th>
                          <th>状态</th>
                          <th>更新时间</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="schedule in inboundSchedules" :key="schedule.schedule_id">
                          <td class="schedule-id">#{{ schedule.schedule_id }}</td>
                          <td class="station-info">
                            <div class="station-name">{{ schedule.start_station.name }}</div>
                            <div class="station-id">{{ schedule.start_station.id }}</div>
                          </td>
                          <td class="bikes-count">{{ schedule.bikes_to_move }} 辆</td>
                          <td>
                            <span class="status-badge" :style="{ backgroundColor: getStatusColor(schedule.status) }">
                              {{ getStatusText(schedule.status) }}
                            </span>
                          </td>
                          <td class="update-time">{{ formatDateTime(schedule.updated_at) }}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                <!-- 调出表格 -->
                <div v-if="outboundSchedules.length > 0" class="dispatch-table-section">
                  <div class="table-header outbound-header">
                    <div class="table-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                        <polyline points="16,17 21,12 16,7"/>
                        <line x1="21" y1="12" x2="9" y2="12"/>
                      </svg>
                    </div>
                    <h5 class="table-title">调出任务 ({{ outboundSchedules.length }})</h5>
                  </div>
                  <div class="table-container">
                    <table class="dispatch-table">
                      <thead>
                        <tr>
                          <th>调度ID</th>
                          <th>目标站点</th>
                          <th>调度数量</th>
                          <th>状态</th>
                          <th>更新时间</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr v-for="schedule in outboundSchedules" :key="schedule.schedule_id">
                          <td class="schedule-id">#{{ schedule.schedule_id }}</td>
                          <td class="station-info">
                            <div class="station-name">{{ schedule.end_station.name }}</div>
                            <div class="station-id">{{ schedule.end_station.id }}</div>
                          </td>
                          <td class="bikes-count">{{ schedule.bikes_to_move }} 辆</td>
                          <td>
                            <span class="status-badge" :style="{ backgroundColor: getStatusColor(schedule.status) }">
                              {{ getStatusText(schedule.status) }}
                            </span>
                          </td>
                          <td class="update-time">{{ formatDateTime(schedule.updated_at) }}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
              
              <!-- 无调度数据 -->
              <div v-else class="no-dispatch-data">
                <div class="no-data-icon">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                    <polyline points="9,22 9,12 15,12 15,22"/>
                  </svg>
                </div>
                <div class="no-data-text">暂无调度任务</div>
                <div class="no-data-desc">该时间段内没有调入或调出任务</div>
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
  max-width: 900px;
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
  margin: 0 0 4px 0;
  opacity: 0.9;
}

.time-subtitle {
  font-size: 12px;
  margin: 0;
  opacity: 0.8;
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

/* 调度概览 */
.dispatch-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-bottom: 32px;
}

.overview-item {
  background: white;
  padding: 24px;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.overview-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
}

.overview-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.1);
}

.overview-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.total-icon {
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
}

.in-icon {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

.out-icon {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
}

.overview-icon svg {
  width: 24px;
  height: 24px;
}

.overview-content {
  flex: 1;
}

.overview-value {
  font-size: 28px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 4px;
}

.overview-label {
  font-size: 14px;
  color: #64748b;
  font-weight: 500;
}

/* 调度表格 */
.dispatch-tables {
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.dispatch-table-section {
  background: white;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
}

.table-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 20px 24px;
  border-bottom: 1px solid #e2e8f0;
}

.inbound-header {
  background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
  color: #065f46;
}

.outbound-header {
  background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
  color: #7f1d1d;
}

.table-icon {
  width: 32px;
  height: 32px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.table-icon svg {
  width: 16px;
  height: 16px;
}

.table-title {
  font-size: 16px;
  font-weight: 600;
  margin: 0;
}

.table-container {
  overflow-x: auto;
}

.dispatch-table {
  width: 100%;
  border-collapse: collapse;
}

.dispatch-table th {
  background: #f8fafc;
  padding: 16px;
  text-align: left;
  font-weight: 600;
  color: #475569;
  font-size: 14px;
  border-bottom: 2px solid #e2e8f0;
}

.dispatch-table td {
  padding: 16px;
  border-bottom: 1px solid #e2e8f0;
  font-size: 14px;
  color: #334155;
}

.dispatch-table tr:hover {
  background: #f8fafc;
}

.schedule-id {
  font-weight: 600;
  color: #4953c2;
  font-family: 'Courier New', monospace;
}

.station-info {
  min-width: 150px;
}

.station-name {
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 4px;
}

.station-id {
  font-size: 12px;
  color: #64748b;
  font-family: 'Courier New', monospace;
}

.bikes-count {
  font-weight: 600;
  color: #059669;
  text-align: center;
}

.status-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  color: white;
  text-align: center;
  min-width: 60px;
}

.update-time {
  font-size: 12px;
  color: #64748b;
  min-width: 120px;
}

/* 无数据状态 */
.no-dispatch-data {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  gap: 16px;
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
  font-size: 18px;
  color: #64748b;
  font-weight: 600;
}

.no-data-desc {
  font-size: 14px;
  color: #94a3b8;
  text-align: center;
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
  
  .dispatch-overview {
    grid-template-columns: 1fr;
  }
  
  .table-container {
    overflow-x: auto;
  }
  
  .dispatch-table {
    min-width: 600px;
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
    width: 40px;
    height: 40px;
  }
  
  .dialog-title {
    font-size: 18px;
  }
  
  .station-subtitle {
    font-size: 13px;
  }
  
  .time-subtitle {
    font-size: 11px;
  }
  
  .overview-value {
    font-size: 24px;
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
  background: #cbd5e1;
  border-radius: 3px;
}
.dialog-content::-webkit-scrollbar-thumb:hover {
  background: #94a3b8;
}
.dialog-content::-webkit-scrollbar-thumb:active {
  background: #6b7280;
}
</style>