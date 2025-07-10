<template>
  <div class="profile-page">
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
      <div class="profile-container">
        <!-- 个人信息标题 -->
        <div class="profile-header">
          <h1>个人信息</h1>
          <p class="profile-subtitle">查看和管理您的个人资料</p>
        </div>

        <!-- 个人信息内容 -->
        <div class="profile-content">
          <div class="profile-section">
            <div class="info-container">
              <h3>基本信息</h3>
              <div class="info-group">
                <div class="info-item">
                  <label class="info-label">账号</label>
                  <div class="info-value">{{ account }}</div>
                </div>
                <div class="info-item">
                  <label class="info-label">邮箱</label>
                  <div class="info-value">{{ email }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const welcoming = ref('管理员，欢迎您！')
const account = ref('')
const email = ref('')

const logout = async () => {
  try {
    await fetch('/api/user/logout', { method: 'POST' })
  } catch (err) {
    console.warn('退出失败', err)
  }
  router.push('/login')
}

onMounted(() => {
  account.value = localStorage.getItem('account') || '未登录'
  email.value = localStorage.getItem('email') || '未绑定'
})
</script>

<style scoped>
.profile-page {
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
  gap: 8px;
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

.profile-container {
  background-color: #fff;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  padding: 40px;
  max-width: 800px;
  width: 100%;
}

.profile-header {
  text-align: center;
  margin-bottom: 40px;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 20px;
}

.profile-header h1 {
  font-size: 28px;
  color: #091275;
  margin: 0 0 10px 0;
  font-weight: 600;
}

.profile-subtitle {
  font-size: 16px;
  color: #6c757d;
  margin: 0;
}

.profile-content {
  display: flex;
  flex-direction: column;
  gap: 30px;
}


/* 信息容器样式 */
.info-container {
  padding: 25px;
  background-color: #fff;
}

.info-container h3 {
  margin: 0 0 20px 0;
  color: #091275;
  font-size: 18px;
  font-weight: 600;
}

.info-group {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.info-item {
  display: flex;
  align-items: center;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 6px;
  border-left: 4px solid #091275;
}

.info-label {
  font-weight: 600;
  color: #495057;
  min-width: 80px;
  margin-right: 20px;
}

.info-value {
  font-size: 16px;
  color: #212529;
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
  .profile-container {
    padding: 20px;
    margin: 0 10px;
  }
  
  .info-item {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .info-label {
    min-width: auto;
    margin-right: 0;
    margin-bottom: 5px;
  }
  
  .profile-header h1 {
    font-size: 24px;
  }
}
</style>