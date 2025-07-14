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
        <!-- 系统设置标题 -->
        <div class="settings-header">
          <h1>系统设置</h1>
          <p class="settings-subtitle">管理您的系统配置和参数</p>
        </div>

        <!-- 设置内容 -->
        <div class="settings-content">
          <div class="settings-section">
            <div class="date-setting-container">
              <h3>日期设置</h3>
              <div class="date-setting">
                <label for="date-input">当前日期：</label>
                <input id="date-input" type="date" v-model="selectedDate" />
                <button class="save-date-button" @click="saveDate">保存日期</button>
                <!-- 内联成功提示 -->
                <div v-if="showInlineSuccess" class="inline-success-toast">
                  <div class="inline-toast-icon">✓</div>
                  <div class="inline-toast-message">{{ inlineSuccessMessage }}</div>
                </div>
              </div>
              <div class="current-date-display">
                <p>已设置日期：{{ displayDate }}</p>
              </div>
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
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    // 清除所有 sessionStorage 项
    sessionStorage.clear()
    router.push('/login')
  }
}

// 读取 localStorage 中的日期，若无则使用今天
const today = new Date().toISOString().split('T')[0]
const selectedDate = ref(localStorage.getItem('selectedDate') || today)

const displayDate = computed(() => {
  if (selectedDate.value) {
    return new Date(selectedDate.value).toLocaleDateString('zh-CN')
  }
  return ''
})

// 显示内联成功提示
const showInlineToast = (message) => {
  inlineSuccessMessage.value = message
  showInlineSuccess.value = true
  
  // 3秒后自动隐藏
  setTimeout(() => {
    showInlineSuccess.value = false
  }, 3000)
}

const saveDate = () => {
  localStorage.setItem('selectedDate', selectedDate.value)
  const formattedDate = new Date(selectedDate.value).toLocaleDateString('zh-CN')
  showInlineToast(`日期已保存为：${formattedDate}`)
}

onMounted(() => {
  // 组件加载时确保 selectedDate 是最新的
  selectedDate.value = localStorage.getItem('selectedDate') || today
})
</script>

<style scoped>
.settings-page {
  min-height: 100vh;
  background-color: #f8f9fa;
  display: flex;
  flex-direction: column;
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

.settings-container {
  background-color: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 40px;
  max-width: 800px;
  width: 100%;
}

.settings-header {
  text-align: center;
  margin-bottom: 40px;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 20px;
}

.settings-header h1 {
  font-size: 28px;
  color: #091275;
  margin: 0 0 10px 0;
  font-weight: 600;
}

.settings-subtitle {
  font-size: 16px;
  color: #6c757d;
  margin: 0;
}

.settings-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

/* 日期设置样式 */
.date-setting-container {
  padding: 25px;
  background-color: #fff;
}

.date-setting-container h3 {
  margin: 0 0 20px 0;
  color: #091275;
  font-size: 18px;
  font-weight: 600;
}

.date-setting {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.date-setting label {
  font-weight: 600;
  min-width: 80px;
  color: #495057;
}

.date-setting input[type="date"] {
  padding: 10px;
  border: 1px solid #ced4da;
  border-radius: 4px;
  font-size: 14px;
  transition: border-color 0.2s;
}

.date-setting input[type="date"]:focus {
  outline: none;
  border-color: #091275;
  box-shadow: 0 0 0 2px rgba(9, 18, 117, 0.2);
}

.save-date-button {
  padding: 10px 20px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  transition: background-color 0.2s;
}

.save-date-button:hover {
  background-color: #0d1c9e;
}

/* 内联成功提示样式 */
.inline-success-toast {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background-color: #d4edda;
  border: 1px solid #c3e6cb;
  border-radius: 4px;
  color: #155724;
  font-size: 14px;
  animation: fadeInScale 0.3s ease-out;
  white-space: nowrap;
}

.inline-toast-icon {
  font-size: 14px;
  font-weight: bold;
  color: #28a745;
}

.inline-toast-message {
  font-size: 13px;
  font-weight: 500;
}

@keyframes fadeInScale {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.current-date-display {
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
  border-left: 4px solid #091275;
}

.current-date-display p {
  margin: 0;
  font-size: 14px;
  color: #495057;
  font-weight: 500;
}

/* 占位符区域 */
.placeholder-section {
  padding: 25px;
  background-color: #f8f9fa;
  text-align: center;
}

.placeholder-section h3 {
  margin: 0 0 15px 0;
  color: #091275;
  font-size: 18px;
  font-weight: 600;
}

.placeholder-section p {
  margin: 0;
  color: #6c757d;
  font-style: italic;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .settings-container {
    padding: 20px;
    margin: 0 10px;
  }
  
  .date-setting {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .date-setting label {
    min-width: auto;
  }
  
  .settings-header h1 {
    font-size: 24px;
  }
  
  .inline-success-toast {
    white-space: normal;
    text-align: center;
  }
}
</style>