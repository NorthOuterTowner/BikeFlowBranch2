<template>
  <div class="profile-page">
    <!-- Header -->
    <header class="header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度系统个人设置</h1>
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
      <div class="profile-container">
        <!-- 基本信息卡片 -->
        <div class="profile-card">
          <div class="card-header">
            <div class="card-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor"/>
              </svg>
            </div>
            <div class="card-title">
              <h3>个人信息</h3>
            </div>
          </div>

          <div class="card-content">
            <div class="info-group">
              <div class="info-item">
                <label class="info-label">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor"/>
                  </svg>
                  账号
                </label>
                <div class="info-value">{{ account }}</div>
              </div>
              <div class="info-item">
                <label class="info-label">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M20 4H4C2.9 4 2.01 4.9 2.01 6L2 18C2 19.1 2.9 20 4 20H20C21.1 20 22 19.1 22 18V6C22 4.9 21.1 4 20 4ZM20 8L12 13L4 8V6L12 11L20 6V8Z" fill="currentColor"/>
                  </svg>
                  邮箱
                </label>
                <div class="info-value">{{ email }}</div>
              </div>
            </div>
          </div>
        </div>

        <!-- 修改账号卡片 -->
        <div class="profile-card">
          <div class="card-header">
            <div class="card-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M3 17.25V21H6.75L17.81 9.94L14.06 6.19L3 17.25ZM20.71 7.04C21.1 6.65 21.1 6.02 20.71 5.63L18.37 3.29C17.98 2.9 17.35 2.9 16.96 3.29L15.13 5.12L18.88 8.87L20.71 7.04Z" fill="currentColor"/>
              </svg>
            </div>
            <div class="card-title">
              <h3>修改账号</h3>
            </div>
          </div>

          <div class="card-content">
            <div class="form-group">
              <label class="form-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="currentColor"/>
                </svg>
                新账号
              </label>
              <input
                type="text"
                v-model="newAccount"
                placeholder="请输入新账号"
                class="form-input"
              />
            </div>

            <div class="form-actions">
              <button class="save-button" @click="handleResetAccount">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M17 3H7C5.89 3 5 3.89 5 5V19C5 20.1 5.89 21 7 21H17C18.1 21 19 20.1 19 19V5C19 3.89 18.1 3 17 3ZM17 19H7V5H10V9H14V5H17V19Z" fill="currentColor"/>
                </svg>
                修改账号
              </button>
            </div>
          </div>
        </div>

        <!-- 修改密码卡片 -->
        <div class="profile-card">
          <div class="card-header">
            <div class="card-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M18 8H17V6C17 3.24 14.76 1 12 1C9.24 1 7 3.24 7 6V8H6C4.9 8 4 8.9 4 10V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V10C20 8.9 19.1 8 18 8ZM12 17C10.9 17 10 16.1 10 15C10 13.9 10.9 13 12 13C13.1 13 14 13.9 14 15C14 16.1 13.1 17 12 17ZM15.1 8H8.9V6C8.9 4.29 10.29 2.9 12 2.9C13.71 2.9 15.1 4.29 15.1 6V8Z" fill="currentColor"/>
              </svg>
            </div>
            <div class="card-title">
              <h3>修改密码</h3>
            </div>
          </div>

          <div class="card-content">
            <div class="form-group">
              <label class="form-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M20 4H4C2.9 4 2.01 4.9 2.01 6L2 18C2 19.1 2.9 20 4 20H20C21.1 20 22 19.1 22 18V6C22 4.9 21.1 4 20 4ZM20 8L12 13L4 8V6L12 11L20 6V8Z" fill="currentColor"/>
                </svg>
                绑定邮箱
              </label>
              <input
                type="email"
                v-model="email"
                placeholder="请输入绑定的邮箱"
                disabled
                class="form-input"
              />
            </div>

            <div class="form-group">
              <label class="form-label">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M18 8H17V6C17 3.24 14.76 1 12 1C9.24 1 7 3.24 7 6V8H6C4.9 8 4 8.9 4 10V20C4 21.1 4.9 22 6 22H18C19.1 22 20 21.1 20 20V10C20 8.9 19.1 8 18 8ZM12 17C10.9 17 10 16.1 10 15C10 13.9 10.9 13 12 13C13.1 13 14 13.9 14 15C14 16.1 13.1 17 12 17ZM15.1 8H8.9V6C8.9 4.29 10.29 2.9 12 2.9C13.71 2.9 15.1 4.29 15.1 6V8Z" fill="currentColor"/>
                </svg>
                新密码
              </label>
              <input
                type="password"
                v-model="newPassword"
                placeholder="请输入新密码"
                class="form-input"
              />
            </div>

            <div class="form-actions">
              <button class="save-button" @click="handleResetPassword">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M17 3H7C5.89 3 5 3.89 5 5V19C5 20.1 5.89 21 7 21H17C18.1 21 19 20.1 19 19V5C19 3.89 18.1 3 17 3ZM17 19H7V5H10V9H14V5H17V19Z" fill="currentColor"/>
                </svg>
                修改密码
              </button>
            </div>
          </div>
        </div>

        <!-- 消息提示 -->
        <Transition name="toast">
          <div v-if="message" class="message-toast">
            <div class="toast-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" fill="currentColor"/>
              </svg>
            </div>
            <div class="toast-message">{{ message }}</div>
          </div>
        </Transition>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import request from '../../api/axios' // 根据实际路径调整

const router = useRouter()
const welcoming = ref('管理员，欢迎您！')
const account = ref('')
const email = ref('')
const newAccount = ref('')
const message = ref('')
const newPassword = ref('')

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

// 调用接口重置账号
async function resetAccount(oldName, newName) {
  return request.post(
    '/reset/account',
    { oldName, newName },
    {
      headers: {
        account: sessionStorage.getItem('account'),
        token: sessionStorage.getItem('token'),
      },
    }
  )
}

async function reserPassword(email, newPassword) {
  return request.post(
    '/reset/pwd',
    { email, newPassword },
    {
      headers: {
        account: sessionStorage.getItem('account'),
        token: sessionStorage.getItem('token'),
      },
    }
  )
}

const handleResetPassword = async () => {
  console.log('handleResetPassword 调用')
  if (!newPassword.value.trim()) {
    showMessage('请输入新密码')
    console.log('新密码为空，退出')
    return
  }
  try {
    const emailValue = sessionStorage.getItem('email')
    console.log('email:', emailValue)
    if (!emailValue) {
      showMessage('当前未绑定邮箱，无法修改密码')
      console.log('未绑定邮箱，退出')
      return
    }
    const res = await reserPassword(emailValue, newPassword.value.trim()) 
    console.log('接口返回:', res)
    if (res.data.status === 200) {
      showMessage(res.data.msg || '请在邮箱确认修改密码')
      newPassword.value = ''
    } else {
      showMessage(res.data.msg || '密码重置失败')
      console.log('接口返回失败:', res.data)
    }
  } catch (error) {
    showMessage('请求失败，请稍后重试')
    console.error('请求异常:', error)
  }
}

const handleResetAccount = async () => {
  console.log('handleResetAccount 调用')
  if (!newAccount.value.trim()) {
    showMessage('请输入新账号')
    console.log('新账号为空，退出')
    return
  }
  try {
    const oldName = sessionStorage.getItem('account')
    console.log('oldName:', oldName)
    if (!oldName) {
      showMessage('当前未登录，无法修改账号')
      console.log('未登录，退出')
      return
    }
    const res = await resetAccount(oldName, newAccount.value.trim())
    console.log('接口返回:', res)
    if (res.data.status === 200) {
      showMessage(res.data.msg || '账号重置成功')
      account.value = newAccount.value.trim()
      sessionStorage.setItem('account', newAccount.value.trim())
      newAccount.value = ''
    } else {
      showMessage(res.data.msg || '账号重置失败')
      console.log('接口返回失败:', res.data)
    }
  } catch (error) {
    showMessage('请求失败，请稍后重试')
    console.error('请求异常:', error)
  }
}

// 显示消息提示
const showMessage = (msg) => {
  message.value = msg
  setTimeout(() => {
    message.value = ''
  }, 3000)
}

onMounted(() => {
  account.value = sessionStorage.getItem('account') || '未登录'
  email.value = sessionStorage.getItem('email') || '未绑定'
})
</script>

<style scoped>
.profile-page {
  min-height: 100vh;
  background: linear-gradient(135deg, #4953c2 0%, #0f1a87 100%);
  display: flex;
  flex-direction: column;
  position: relative;
  overflow-y: auto;
}

.profile-page::before {
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
  justify-content: space-between;
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
  overflow-y: auto;
  max-height: calc(100vh - 80px); /* 减去header高度 */
}

.profile-container {
  max-width: 800px;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding-bottom: 40px; /* 底部留出空间 */
}

/* 个人信息卡片 */
.profile-card {
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
  color: black;
}

.card-content {
  padding: 32px;
}

/* 信息展示样式 */
.info-group {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.info-item {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  background: #f8f9fa;
  border-radius: 12px;
  border-left: 4px solid #4953c2;
}

.info-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: #2d3748;
  min-width: 80px;
  margin-right: 20px;
  font-size: 14px;
}

.info-value {
  font-size: 16px;
  color: #2d3748;
  font-weight: 500;
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

.form-input {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  font-size: 14px;
  transition: all 0.3s ease;
  background: white;
  color: #2d3748;
}

.form-input:focus {
  outline: none;
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  transform: translateY(-1px);
}

.form-input:disabled {
  background-color: #f8f9fa;
  color: #6c757d;
  cursor: not-allowed;
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

/* 消息提示 */
.message-toast {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
  background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
  color: white;
  border-radius: 12px;
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
  .header {
    padding: 16px 20px;
  }
  
  .main-content {
    padding: 20px 16px;
    max-height: calc(100vh - 70px); /* 移动端调整header高度 */
  }
  
  .card-content {
    padding: 24px 20px;
  }
  
  .card-header {
    padding: 20px;
  }
  
  .profile-container {
    gap: 16px;
    padding-bottom: 20px;
  }
  
  .info-item {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .info-label {
    min-width: auto;
    margin-right: 0;
    margin-bottom: 8px;
  }
  
  .form-actions {
    justify-content: center;
  }
  
  .save-button {
    width: 100%;
    justify-content: center;
  }
}
.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
}
</style>