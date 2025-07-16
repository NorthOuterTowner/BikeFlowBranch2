<template>
  <div class="settings-page">
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
      <div class="settings-container">
        <!-- 设置卡片 -->
        <div class="settings-card">
          <div class="card-header">
            <div class="card-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
              </svg>
            </div>
            <div class="card-title">
              <h3>日期时间设置</h3>
            </div>
          </div>

          <div class="card-content">
            <div class="form-group">
              <label for="date-input" class="form-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M19 3H18V1H16V3H8V1H6V3H5C3.89 3 3 3.89 3 5V19C3 20.1 3.89 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.89 20.1 3 19 3ZM19 19H5V8H19V19Z" fill="currentColor"/>
                </svg>
                当前日期
              </label>
              <input 
                id="date-input" 
                type="date" 
                v-model="selectedDate" 
                class="form-input"
              />
            </div>

            <div class="form-group">
              <label for="hour-select" class="form-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67V7z" fill="currentColor"/>
                </svg>
                当前时间
              </label>
              <select id="hour-select" v-model="selectedHour" class="form-select">
                <option v-for="h in 24" :key="h" :value="(h < 10 ? '0' + h : h) + ':00'">
                  {{ (h < 10 ? '0' + h : h) + ':00' }}
                </option>
              </select>
            </div>

            <div class="form-actions">
              <button class="save-button" @click="saveDateTime">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M17 3H7C5.89 3 5 3.89 5 5V19C5 20.1 5.89 21 7 21H17C18.1 21 19 20.1 19 19V5C19 3.89 18.1 3 17 3ZM17 19H7V5H10V9H14V5H17V19Z" fill="currentColor"/>
                </svg>
                保存设置
              </button>
            </div>

            <!-- 内联成功提示 -->
            <Transition name="toast">
              <div v-if="showInlineSuccess" class="success-toast">
                <div class="toast-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z" fill="currentColor"/>
                  </svg>
                </div>
                <div class="toast-message">{{ inlineSuccessMessage }}</div>
              </div>
            </Transition>
          </div>
        </div>

        <!-- 当前设置显示 -->
        <div class="status-card">
          <div class="status-header">
            <h4>当前设置</h4>
          </div>
          <div class="status-content">
            <div class="status-item">
              <span class="status-label">日期：</span>
              <span class="status-value">{{ displayDate }}</span>
            </div>
            <div class="status-item">
              <span class="status-label">时间：</span>
              <span class="status-value">{{ displayHour }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const welcoming = ref('管理员，欢迎您！')
const showInlineSuccess = ref(false)
const inlineSuccessMessage = ref('')

const logout = async () => {
  const confirmed = window.confirm('确定要退出登录吗？')
  if (!confirmed) {
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

// 读取 localStorage 中的日期，若无则使用今天
const today = new Date().toISOString().split('T')[0]
const selectedDate = ref(localStorage.getItem('selectedDate') || today)
const selectedHour = ref(localStorage.getItem('selectedHour') || '09:00')

const displayHour = computed(() => selectedHour.value)
const displayDate = computed(() => {
  const date = new Date(selectedDate.value)
  return date.toLocaleDateString('zh-CN')
})

const saveDateTime = () => {
  localStorage.setItem('selectedDate', selectedDate.value)
  localStorage.setItem('selectedHour', selectedHour.value)

  const formattedDate = new Date(selectedDate.value).toLocaleDateString('zh-CN')
  showInlineToast(`日期时间已保存为：${formattedDate} ${selectedHour.value}`)
}

// 显示内联成功提示
const showInlineToast = (message) => {
  inlineSuccessMessage.value = message
  showInlineSuccess.value = true
  
  // 3秒后自动隐藏
  setTimeout(() => {
    showInlineSuccess.value = false
  }, 3000)
}

onMounted(() => {
  selectedDate.value = localStorage.getItem('selectedDate') || today
})
</script>

<style scoped>
.settings-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  display: flex;
  flex-direction: column;
  position: relative;
}

.settings-page::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
  pointer-events: none;
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
  position: relative;
  z-index: 1;
}

.settings-container {
  max-width: 800px;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* 设置卡片 */
.settings-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 24px 32px;
  background: linear-gradient(135deg, #f2f2f5 0%, #ebe8ed 100%);
  color: white;
}

.card-icon {
  width: 48px;
  height: 48px;
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
}

.card-title h3 {
  margin: 0 0 4px 0;
  font-size: 20px;
  font-weight: 600;
  color:black;
}

.card-content {
  padding: 32px;
}

/* 表单样式 */
.form-group {
  margin-bottom: 24px;
}

.form-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: #2d3748;
  margin-bottom: 8px;
  font-size: 14px;
}

.form-input, .form-select {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  font-size: 14px;
  transition: all 0.3s ease;
  background: white;
  color: #2d3748;
}

.form-input:focus, .form-select:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  transform: translateY(-1px);
}

.form-select {
  cursor: pointer;
}

.form-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 32px;
}

.save-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 24px;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  color: white;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
}

.save-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
}

/* 成功提示 */
.success-toast {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
  color: white;
  border-radius: 12px;
  margin-top: 16px;
  box-shadow: 0 4px 20px rgba(72, 187, 120, 0.3);
}

.toast-icon {
  width: 20px;
  height: 20px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.toast-message {
  font-size: 14px;
  font-weight: 500;
}

/* 状态卡片 */
.status-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 24px;
}

.status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.status-header h4 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #2d3748;
}


@keyframes pulse {
  0% { opacity: 1; }
  50% { opacity: 0.5; }
  100% { opacity: 1; }
}

.status-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid #e2e8f0;
}

.status-item:last-child {
  border-bottom: none;
}

.status-label {
  font-weight: 500;
  color: #4a5568;
  font-size: 14px;
}

.status-value {
  font-weight: 600;
  color: #2d3748;
  font-size: 14px;
}

/* 动画 */
.toast-enter-active, .toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from {
  opacity: 0;
  transform: translateY(-10px) scale(0.95);
}

.toast-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(0.95);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-content {
    padding: 16px 20px;
    flex-direction: column;
    gap: 16px;
  }
  
  .main-content {
    padding: 20px 16px;
  }
  
  .card-content {
    padding: 24px 20px;
  }
  
  .card-header {
    padding: 20px;
  }
  
  .settings-container {
    gap: 16px;
  }
  
  .form-actions {
    justify-content: center;
  }
  
  .save-button {
    width: 100%;
    justify-content: center;
  }
}
</style>