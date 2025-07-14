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

              <!-- 新增：修改账号 -->
              <div class="reset-account">
                <h3>修改账号</h3>
                <input
                  type="text"
                  v-model="newAccount"
                  placeholder="请输入新账号"
                />
                <button @click="handleResetAccount">修改账号</button>
              </div>

              <!-- 新增：修改密码 -->
              <div class="reset-account">
                <h3>修改密码</h3>
                <input
                  type="email"
                  v-model="email"
                  placeholder="请输入绑定的邮箱"
                  disabled
                />
                <input
                  type="password"
                  v-model="newPassword"
                  placeholder="请输入新密码"
                />
                <button @click="handleResetPassword">修改密码</button>
              </div>

              <div v-if="message" class="message">{{ message }}</div>
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
import request from '../../api/axios' // 根据实际路径调整

const router = useRouter()
const welcoming = ref('管理员，欢迎您！')
const account = ref('')
const email = ref('')
const newAccount = ref('')
const message = ref('')

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
  if (!newAccount.value.trim()) {
    message.value = '请输入新密码'
    console.log('新密码为空，退出')
    return
  }
  try {
    const emailValue = sessionStorage.getItem('email')
    console.log('email:', emailValue)
    if (!emailValue) {
      message.value = '当前未绑定邮箱，无法修改密码'
      console.log('未绑定邮箱，退出')
      return
    }
    const res = await reserPassword(emailValue, newAccount.value.trim())
    console.log('接口返回:', res)
    if (res.data.status === 200) {
      message.value = res.data.msg || '请在邮箱确认修改密码'
      newAccount.value = ''
    } else {
      message.value = res.data.msg || '密码重置失败'
      console.log('接口返回失败:', res.data)
    }
  } catch (error) {
    message.value = '请求失败，请稍后重试'
    console.error('请求异常:', error)
  }
}

const handleResetAccount = async () => {
  console.log('handleResetAccount 调用')
  if (!newAccount.value.trim()) {
    message.value = '请输入新账号'
    console.log('新账号为空，退出')
    return
  }
  try {
    const oldName = sessionStorage.getItem('account')
    console.log('oldName:', oldName)
    if (!oldName) {
      message.value = '当前未登录，无法修改账号'
      console.log('未登录，退出')
      return
    }
    const res = await resetAccount(oldName, newAccount.value.trim())
    console.log('接口返回:', res)
    if (res.data.status === 200) {
      message.value = res.data.msg || '账号重置成功'
      account.value = newAccount.value.trim()
      sessionStorage.setItem('account', newAccount.value.trim())
      newAccount.value = ''
    } else {
      message.value = res.data.msg || '账号重置失败'
      console.log('接口返回失败:', res.data)
}

  } catch (error) {
    message.value = '请求失败，请稍后重试'
    console.error('请求异常:', error)
  }
}



onMounted(() => {
  account.value = sessionStorage.getItem('account') || '未登录'
  email.value = sessionStorage.getItem('email') || '未绑定'
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

.reset-account {
  margin-top: 20px;
}
.reset-account input {
  padding: 6px 8px;
  width: 200px;
  margin-right: 10px;
}
.reset-account button {
   padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
}
.reset-account button:hover {
  background-color: #0d1c9e;
}

.message {
  margin-top: 10px;
  color: rgb(0, 0, 0);
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