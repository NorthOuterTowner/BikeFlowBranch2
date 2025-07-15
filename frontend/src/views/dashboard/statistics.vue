<template>
  <div class="statics-page">
    <!-- Header -->
    <header class="header">
      <div class="user-info">
        <div class="user-top">
          <span class="welcoming">{{ welcoming }}</span>
          <button class="logout-button" @click="logout">退出</button>
        </div>
      </div>
    </header>

    <!-- 主内容区域 -->
    <div class="main-content">
      <div class="statics-container">
        <!-- 时间筛选器 -->
        <div class="filter-section">
          <h2>流量统计</h2>
          <div class="date-filter">
            <div class="date-item">
              <label>开始日期:</label>
              <input type="date" v-model="startDate" />
            </div>
            <div class="date-item">
              <label>开始小时:</label>
              <select v-model="startHour">
                <option v-for="hour in 24" :key="hour-1" :value="(hour-1).toString().padStart(2, '0')">
                  {{ (hour-1).toString().padStart(2, '0') }}:00
                </option>
              </select>
            </div>
            <div class="date-item">
              <label>结束日期:</label>
              <input type="date" v-model="endDate" />
            </div>
            <div class="date-item">
              <label>结束小时:</label>
              <select v-model="endHour">
                <option v-for="hour in 24" :key="hour-1" :value="(hour-1).toString().padStart(2, '0')">
                  {{ (hour-1).toString().padStart(2, '0') }}:00
                </option>
              </select>
            </div>
            <button class="query-button" @click="fetchTop10Data" :disabled="loading">
              {{ loading ? '查询中...' : '查询' }}
            </button>
          </div>
        </div>

        <!-- 图表区域 -->
        <div class="chart-section">
          <div class="chart-container">
            <canvas ref="chartCanvas" width="800" height="400"></canvas>
          </div>
        </div>

        <div class="line-chart-section" style="width: 800px; height: 400px; margin: 20px auto;">
          <canvas ref="lineCanvas" width="800" height="400"></canvas>
        </div>

        <!-- 折叠按钮 -->
        <div v-if="chartData.length > 0" class="toggle-table-button">
          <button @click="showTable = !showTable">
            {{ showTable ? '隐藏详细数据' : '展开详细数据' }}
          </button>
        </div>

        <!-- 数据表格 -->
        <div class="table-section" v-if="chartData.length > 0 && showTable">
          <h3>详细数据</h3>
          <table class="data-table">
            <thead>
              <tr>
                <th>排名</th>
                <th>站点ID</th>
                <th>总流量</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(item, index) in chartData" :key="item.station_id">
                <td>{{ index + 1 }}</td>
                <td>{{ item.station_id }}</td>
                <td>{{ item.total_flow }}</td>
              </tr>
            </tbody>
          </table>
        </div>

      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import request from '@/api/axios'

const router = useRouter()
const welcoming = ref('管理员，欢迎您！')

// 数据相关
const chartData = ref([])
const loading = ref(false)
const chartCanvas = ref(null)
let chartInstance = null

// 日期时间筛选
const startDate = ref('')
const startHour = ref('00')
const endDate = ref('')
const endHour = ref('23')

const initDefaultDates = () => {
  const today = new Date()
  const yesterday = new Date(today)
  yesterday.setDate(today.getDate() - 1)

  const savedDate = localStorage.getItem('selectedDate') // 用作统一的日期来源

  // 如果有保存的 selectedDate，则同时作为开始和结束日期
  if (savedDate) {
    startDate.value = savedDate
    endDate.value = savedDate
  } else {
    // 否则按默认：昨天作为开始，今天作为结束
    startDate.value = yesterday.toISOString().split('T')[0]
    endDate.value = today.toISOString().split('T')[0]
  }

  // 小时默认不变
  startHour.value = '00'
  endHour.value = '23'
}

const showTable = ref(false)

const flowData = ref({ inflow: 0, outflow: 0, total: 0 })
const pieCanvas = ref(null)
let pieChartInstance = null

const fetchFlowSummaryByHour = async (dateStr, hourStr) => {
  try {
    const res = await request.post('/statistics/flow/time', {
      startDate: dateStr,
      startHour: hourStr,
      endDate: dateStr,
      endHour: hourStr
    })
    if (res.data.code === 200) {
      return res.data.data?.total || 0 // 返回该小时总流量，接口需返回 total 字段
    } else {
      console.warn('获取小时流量失败', dateStr, hourStr)
      return 0
    }
  } catch (e) {
    console.error('获取小时流量异常', e)
    return 0
  }
}

const generateHourRange = (startDateStr, startHourStr, endDateStr, endHourStr) => {
  const start = new Date(`${startDateStr}T${startHourStr}:00:00`)
  const end = new Date(`${endDateStr}T${endHourStr}:00:00`)
  const hours = []

  let cur = new Date(start)
  while (cur <= end) {
    const y = cur.getFullYear()
    const m = (cur.getMonth() + 1).toString().padStart(2, '0')
    const d = cur.getDate().toString().padStart(2, '0')
    const h = cur.getHours().toString().padStart(2, '0')

    hours.push([`${y}-${m}-${d}`, h])
    cur.setHours(cur.getHours() + 1)
  }

  return hours
}

const hourlyFlowData = ref([]) // [{ time: '07-14 00:00', flow: 123 }, ...]

const fetchHourlyFlowData = async () => {
  const hourList = generateHourRange(startDate.value, startHour.value, endDate.value, endHour.value)
  hourlyFlowData.value = []
  loading.value = true

  for (const [dateStr, hourStr] of hourList) {
    const flow = await fetchFlowSummaryByHour(dateStr, hourStr)
    hourlyFlowData.value.push({
      time: `${dateStr.slice(5)} ${hourStr}:00`, // MM-DD HH:00格式
      flow
    })
  }

  loading.value = false
  await nextTick()
  renderLineChart()
}
const lineCanvas = ref(null)
let lineChartInstance = null

const renderLineChart = () => {
  if (!lineCanvas.value || hourlyFlowData.value.length === 0) return
  const ctx = lineCanvas.value.getContext('2d')
  if (lineChartInstance) {
    lineChartInstance.destroy()
  }

  const labels = hourlyFlowData.value.map(item => item.time)
  const data = hourlyFlowData.value.map(item => item.flow)

  lineChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: '每小时总流量',
        data,
        borderColor: '#4A90E2',
        backgroundColor: 'rgba(74, 144, 226, 0.3)',
        fill: true,
        tension: 0.3,
        pointRadius: 3,
        pointHoverRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: '时间段内每小时总流量折线图',
          font: { size: 18, weight: 'bold' },
          padding: 20
        },
        tooltip: {
          callbacks: {
            label: ctx => `流量: ${ctx.parsed.y.toLocaleString()}`
          }
        },
        legend: { display: true }
      },
      scales: {
        x: {
          title: { display: true, text: '时间' }
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: '流量' },
          ticks: {
            callback: value => value.toLocaleString()
          }
        }
      }
    }
  })
}


// 获取Top10数据
const fetchTop10Data = async () => {
  if (!startDate.value || !endDate.value) {
    alert('请选择开始和结束日期')
    return
  }

  loading.value = true
  try {
    const response = await request.post('/statistics/top', {
      startDate: startDate.value,
      startHour: startHour.value,
      endDate: endDate.value,
      endHour: endHour.value
    })

    if (response.data.code === 200) {
      chartData.value = response.data.data || []
      await nextTick()
      renderChart()
      // 这里调用按小时流量折线图数据加载
      await fetchHourlyFlowData()
    } else {
      alert('获取数据失败')
    }
  } catch (error) {
    console.error('获取Top10数据失败:', error)
    alert('获取数据失败，请重试')
  } finally {
    loading.value = false
  }
}

// 渲染图表
const renderChart = () => {
  if (!chartCanvas.value || chartData.value.length === 0) return

  const ctx = chartCanvas.value.getContext('2d')
  
  // 销毁之前的图表实例
  if (chartInstance) {
    chartInstance.destroy()
  }

  // 准备图表数据
  const labels = chartData.value.map(item => item.station_id)
  const data = chartData.value.map(item => parseInt(item.total_flow))
  
  // 创建渐变色
  const gradient = ctx.createLinearGradient(0, 0, 0, 400)
  gradient.addColorStop(0, '#4A90E2')
  gradient.addColorStop(1, '#7BB3F0')

  chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: '流量',
        data: data,
        backgroundColor: gradient,
        borderColor: '#4A90E2',
        borderWidth: 1,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: 'Top10站点流量统计',
          font: {
            size: 18,
            weight: 'bold'
          },
          padding: 20
        },
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `流量: ${context.parsed.y.toLocaleString()}`
            }
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: '流量'
          },
          ticks: {
            callback: function(value) {
              return value.toLocaleString()
            }
          }
        },
        x: {
          title: {
            display: true,
            text: '站点ID'
          }
        }
      },
      animation: {
        duration: 1000,
        easing: 'easeInOutQuart'
      }
    }
  })
}

// 登出功能
const logout = async () => {
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    sessionStorage.clear()
    router.push('/login')
  }
}

// 组件挂载
onMounted(() => {
  initDefaultDates()
  
  // 动态加载Chart.js
  if (typeof Chart === 'undefined') {
    const script = document.createElement('script')
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js'
    script.onload = () => {
      fetchTop10Data()
    }
    document.head.appendChild(script)
  } else {
    fetchTop10Data()
  }
})
</script>

<style scoped>
.statics-page {
  min-height: 100vh;
  background-color: #f8f9fa;
  display: flex;
  flex-direction: column;
  /* 添加下面两行 */
  overflow-y: auto;
  max-height: 100vh;
}

/* Header 样式 */
.header {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 10px 20px;
  background-color: #fff;
  border-bottom: 1px solid #e9ecef;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.welcoming {
  font-size: 14px;
  white-space: nowrap;
  color: #495057;
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

.logout-button {
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
}

.logout-button:hover {
  background-color: #0d1c9e;
}

/* 主内容区域 */
.main-content {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 40px 20px;
}

.statics-container {
  width: 100%;
  max-width: 1200px;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  padding: 30px;
}

/* 筛选区域 */
.filter-section {
  margin-bottom: 30px;
}

.filter-section h2 {
  margin-bottom: 20px;
  color: #333;
  font-size: 24px;
}

.date-filter {
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}

.date-item {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.date-item label {
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

.date-item input,
.date-item select {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 4px;
  font-size: 14px;
}

.query-button {
  padding: 8px 20px;
  background-color: #4A90E2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
  margin-top: 20px;
}

.query-button:hover:not(:disabled) {
  background-color: #357ABD;
}

.query-button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

/* 图表区域 */
.chart-section {
  margin-bottom: 30px;
}

.chart-container {
  width: 100%;
  height: 400px;
  border: 1px solid #e9ecef;
  border-radius: 8px;
  padding: 20px;
  background-color: #fafafa;
}

/* 表格区域 */
.table-section {
  margin-top: 30px;
}

.table-section h3 {
  margin-bottom: 15px;
  color: #333;
  font-size: 18px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  background-color: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.data-table th,
.data-table td {
  padding: 12px 15px;
  text-align: left;
  border-bottom: 1px solid #e9ecef;
}

.data-table th {
  background-color: #f8f9fa;
  font-weight: 600;
  color: #495057;
}

.data-table tr:hover {
  background-color: #f8f9fa;
}

.data-table tr:nth-child(even) {
  background-color: #fdfdfd;
}
.toggle-table-button {
  text-align: right;
  margin: 10px 0;
}

.toggle-table-button button {
  background-color: #4A90E2;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
}

.toggle-table-button button:hover {
  background-color: #357ABD;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .date-filter {
    flex-direction: column;
    align-items: stretch;
  }
  
  .date-item {
    width: 100%;
  }
  
  .chart-container {
    height: 300px;
    padding: 10px;
  }
  
  .data-table {
    font-size: 14px;
  }
  
  .data-table th,
  .data-table td {
    padding: 8px 10px;
  }
}
</style>