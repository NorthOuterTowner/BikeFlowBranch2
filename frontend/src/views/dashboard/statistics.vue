<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度数据统计</h1>
      </div>
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

           <!-- 历史流量折线图 -->
        <div class="history-chart-section" style="width: 800px; height: 400px; margin: 20px auto;">
          <canvas ref="historyCanvas" width="800" height="400"></canvas>
        </div>
        
          <div class="date-filter">
            <div class="date-item">
              <label>开始日期:</label>
              <input type="date" v-model="startDate" :max="maxDate" />
            </div>

            <div class="date-item">
              <label>开始小时:</label>
              <select v-model="startHour">
                <option v-for="hour in startHourOptions" :key="hour" :value="hour">
                  {{ hour }}:00
                </option>
              </select>
            </div>

            <div class="date-item">
              <label>结束日期:</label>
              <input type="date" v-model="endDate" :max="maxDate" />
            </div>
            <div class="date-item">
              <label>结束小时:</label>
              <select v-model="endHour">
                <option v-for="hour in endHourOptions" :key="hour" :value="hour">
                  {{ hour }}:00
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

        <!-- 小时流量折线图 -->
        <div class="line-chart-section" style="width: 800px; height: 400px; margin: 20px auto;">
          <canvas ref="lineCanvas" width="800" height="400"></canvas>
        </div>


      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
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

// 从 localStorage 获取最大允许的日期和小时
const maxDateStr = localStorage.getItem('selectedDate') || '' // 格式 YYYY-MM-DD
const maxHourStr = localStorage.getItem('selectedHour') || '23' // 格式 HH，如 "18"

// 转成Date对象方便比较
const maxDateTime = maxDateStr
  ? new Date(`${maxDateStr}T${maxHourStr}:00:00`)
  : new Date() // 如果没设置就用当前时间

// maxDate 用于日期输入框的 max 属性
const maxDate = maxDateStr || new Date().toISOString().slice(0, 10)

// 用于动态生成小时选项，限制小时最大值
const generateHourOptions = (dateStr) => {
  let maxHour = 23
  if (dateStr === maxDateStr) {
    maxHour = parseInt(maxHourStr)
  }
  const hours = []
  for (let h = 0; h <= maxHour; h++) {
    hours.push(h.toString().padStart(2, '0'))
  }
  return hours
}

// 响应式小时选项
const startHourOptions = ref([])
const endHourOptions = ref([])

// 初始化日期和小时
const initDefaultDates = () => {
  if (maxDateStr) {
    startDate.value = maxDateStr
    endDate.value = maxDateStr
  } else {
    startDate.value = maxDate
    endDate.value = maxDate
  }
  startHour.value = maxHourStr.padStart(2, '0')
  endHour.value = maxHourStr.padStart(2, '0')
}

const showTable = ref(false)

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

let lineChartInstance = null // 声明折线图实例
let historyChartInstance = null // 声明历史流量图实例

const lineCanvas = ref(null)
const historyCanvas = ref(null)

const renderLineChart = () => {
  if (!lineCanvas.value || hourlyFlowData.value.length === 0) return

  const ctx = lineCanvas.value.getContext('2d')

  // 销毁旧的实例避免内存泄露
  if (lineChartInstance) {
    lineChartInstance.destroy()
  }

  const labels = hourlyFlowData.value.map(item => item.time)
  const data = hourlyFlowData.value.map(item => item.flow)

  lineChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: '小时流量',
        data: data,
        fill: true,
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: '#4A90E2',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: '小时流量趋势图',
          font: {
            size: 18,
            weight: 'bold'
          },
          padding: 20
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `流量: ${context.parsed.y}`
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
          }
        },
        x: {
          title: {
            display: true,
            text: '时间'
          }
        }
      },
      animation: {
        duration: 800,
        easing: 'easeInOutQuart'
      }
    }
  })
}

const renderHistoryChart = () => {
  console.log('开始渲染历史流量图表')
  console.log('historyCanvas.value:', historyCanvas.value)
  console.log('historyFlowData.value:', historyFlowData.value)
  
  if (!historyCanvas.value || historyFlowData.value.length === 0) {
    console.log('无法渲染历史图表：canvas或数据为空')
    return
  }

  const ctx = historyCanvas.value.getContext('2d')

  // 销毁旧的实例避免内存泄露
  if (historyChartInstance) {
    historyChartInstance.destroy()
  }

  const labels = historyFlowData.value.map(item => item.date)
  const data = historyFlowData.value.map(item => item.total_flow)
  
  console.log('图表标签:', labels)
  console.log('图表数据:', data)

  historyChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: '历史流量',
        data: data,
        fill: true,
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: '#FF6384',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        title: {
          display: true,
          text: '历史流量趋势图',
          font: {
            size: 18,
            weight: 'bold'
          },
          padding: 20
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
            text: '日期'
          }
        }
      },
      animation: {
        duration: 1000,
        easing: 'easeInOutQuart'
      }
    }
  })
  
  console.log('历史流量图表渲染完成')
}

const hourlyFlowData = ref([]) // [{ time: '07-14 00:00', flow: 123 }, ...]
const historyFlowData = ref([]) // [{ date: '2025-01-15', total_flow: 3771 }, ...]

const fetchHourlyFlowDataByDayAPI = async (dateStr) => {
  try {
    const res = await request.get('/statistics/flow/day', {
      params: { query_date: dateStr }
    })
    return res.data.hourly_flows || []
  } catch (e) {
    console.error(`获取 ${dateStr} 小时流量失败:`, e)
    return []
  }
}

const fetchHistoryFlowDataAPI = async (dateStr) => {
  try {
    const res = await request.get('/statistics/flow/days', {
      params: { target_date: dateStr }
    })
    return res.data.daily_summary || []
  } catch (e) {
    console.error(`获取历史流量数据失败:`, e)
    return []
  }
}

const fetchHourlyFlowData = async () => {
  const hourList = generateHourRange(startDate.value, startHour.value, endDate.value, endHour.value)
  const dateSet = new Set(hourList.map(([date]) => date)) // 所有查询涉及的日期
  loading.value = true
  hourlyFlowData.value = []

  const dateToHourlyMap = {} // date -> [24小时流量]

  for (const dateStr of dateSet) {
    const flows = await fetchHourlyFlowDataByDayAPI(dateStr)
    dateToHourlyMap[dateStr] = flows
    await new Promise(r => setTimeout(r, 200)) // 节流
  }

  for (const [dateStr, hourStr] of hourList) {
    const hour = parseInt(hourStr)
    const flowObj = dateToHourlyMap[dateStr]?.find(h => h.hour === hour)
    const flow = flowObj?.total_flow || 0

    hourlyFlowData.value.push({
      time: `${dateStr.slice(5)} ${hourStr}:00`,
      flow
    })
  }

  loading.value = false
  await nextTick()
  renderLineChart()
}

const fetchHistoryFlowData = async () => {
  let selectedDate = localStorage.getItem('selectedDate')

  if (!selectedDate) {
    console.warn('localStorage中没有selectedDate，使用当前日期作为默认值')
    const today = new Date().toISOString().slice(0, 10)
    selectedDate = today
    localStorage.setItem('selectedDate', today)  // ← 补上写入
  }

  try {
    const data = await fetchHistoryFlowDataAPI(selectedDate)
    historyFlowData.value = data
    await nextTick()
    renderHistoryChart()
  } catch (error) {
    console.error('获取历史流量数据失败:', error)
  }
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

// 监听日期变化，动态调整小时选项，并校正小时
watch(startDate, (newDate) => {
  startHourOptions.value = generateHourOptions(newDate)
  if (!startHourOptions.value.includes(startHour.value)) {
    startHour.value = startHourOptions.value[startHourOptions.value.length - 1]
  }
})

watch(endDate, (newDate) => {
  endHourOptions.value = generateHourOptions(newDate)
  if (!endHourOptions.value.includes(endHour.value)) {
    endHour.value = endHourOptions.value[endHourOptions.value.length - 1]
  }
})

// 防止时间超过最大限制，自动修正
watch([startDate, startHour], ([d, h]) => {
  const current = new Date(`${d}T${h}:00:00`)
  if (current > maxDateTime) {
    startDate.value = maxDateStr
    startHour.value = maxHourStr.padStart(2, '0')
  }
})

watch([endDate, endHour], ([d, h]) => {
  const current = new Date(`${d}T${h}:00:00`)
  if (current > maxDateTime) {
    endDate.value = maxDateStr
    endHour.value = maxHourStr.padStart(2, '0')
  }
})

initDefaultDates()

onMounted(async () => {
  startHourOptions.value = generateHourOptions(startDate.value)
  endHourOptions.value = generateHourOptions(endDate.value)

  if (typeof Chart === 'undefined') {
    const script = document.createElement('script')
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js'
    script.onload = async () => {
      await fetchTop10Data()
      await fetchHistoryFlowData()
    }
    document.head.appendChild(script)
  } else {
    await fetchTop10Data()
    await fetchHistoryFlowData()
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
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.main-content {
  flex: 1 1 auto;
  overflow-y: auto; /* 让主内容区域可以滚动 */
  padding: 10px;
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
.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
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
.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-left: 20px;
  gap: 15px;
  flex-shrink: 0;
}
.logout-button { background:#091275; color:white; border:none; padding:4px 8px; border-radius:4px; cursor:pointer; }

</style>