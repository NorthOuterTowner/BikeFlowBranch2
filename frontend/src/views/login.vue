<template>
  <div class="login-page" @keydown.enter="doLogin">
    <n-spin :show="loading">
      <n-card class="login-card" bordered hoverable>
        <h1>BikeFlow</h1>

        <n-input v-model:value="account" placeholder="用户名" clearable size="large" />
        <n-input v-model:value="password" type="password" placeholder="密码" clearable size="large" style="margin-top: 12px;" />

        <n-button type="primary" class="login-button" size="large" style="margin-top: 16px; width: 100%;" @click="doLogin" :disabled="loading">
          登录
        </n-button>

        <p class="to-register">
          没有账号？
          <router-link to="/register">去注册</router-link>
        </p>

        <n-alert v-if="messageStore.message" :type="messageStore.type" style="margin-top: 16px; width: 100%;">
          {{ messageStore.message }}
        </n-alert>
      </n-card>
    </n-spin>

  </div>
</template>
  
<script setup>
  import { ref } from 'vue'
  import { useRouter } from 'vue-router'
  import { login } from '../api/axios' 
  import { useMessageStore } from '../store/messageStore'  // 根据实际路径调整
  import { NInput, NButton, NAlert, NSpin } from 'naive-ui'
  import { onMounted } from 'vue'

  const account = ref('')
  const password = ref('')
  const router = useRouter()
  const messageStore = useMessageStore()
  const loading = ref(false)
  
  onMounted(() => {
   messageStore.setMessage('', '')  // 页面加载时清空提示
  })

  async function doLogin() {
    messageStore.setMessage('', '') // 清空提示
    try {
      const res = await login(account.value, password.value)

      if (res.data.code === 200) {
        messageStore.setMessage('登录成功', 'success')

      const user = res.data.data

      localStorage.setItem('token', user.token)
      localStorage.setItem('account', user.account)
      localStorage.setItem('email', user.email || '')

        loading.value = true // 开始加载状态
        // 等 1.5 秒后再跳转，让用户看到提示
        setTimeout(() => {
          loading.value = false // 结束加载状态
          router.push('/dashboard')
        }, 1000)
      } else {
        messageStore.setMessage('登录失败: ' + res.data.msg, 'error')
      }
    } catch (e) {
      console.error(e)
      console.error('后端返回：', e.response?.data) // 优先取后端返回的错误提示
      const backendMsg = e.response?.data?.msg || e.response?.data?.error

      if (backendMsg) {
        messageStore.setMessage(backendMsg, 'error')
      } else {
        messageStore.setMessage('请求失败，请检查网络或后端服务', 'error')
      }
    }
  }
</script>
  
<style scoped>

.login-card {
  width: 100%;
  width: 400px;
  padding: 32px 24px;
  text-align: center;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.2);
  background-color: #ffffffee; /* 半透明白 */
}
.logo {
  width: 64px;
  height: 64px;
  margin-bottom: 16px;
}
h1 {
  margin-bottom: 24px;
  font-size: 24px;
  font-weight: bold;
  background: linear-gradient(135deg, #ffffff, #0556a7);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.to-register {
  margin-top: 12px;
  font-size: 14px;
}
.to-register a {
  color:linear-gradient(135deg, #ffffff, #0556a7);
  text-decoration: none;
}
.login-button {
  margin-top: 16px;
  width: 100%;
  background: linear-gradient(135deg, #ffffff, #0556a7);
  color: #fff;
  border: none;
}
:deep(.n-input__input) {
  text-align: left;
}

.login-page {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;      
  justify-content: center;  
  height: 100vh;
  /* max-width: 300px; */
  width: 90%;              
  margin: 0 auto;
  text-align: center;
  padding: 0 16px;  
  z-index: 1;
  overflow: hidden;
}

.login-page::before {
  content: "";
  position: absolute;
  top: 0; left: 0;
  width: 100vw;
  height: 100vh;
  background-image: url('../../public/background.png'); /* public 目录下 */
  background-size: cover;
  background-position: center;
  opacity: 0.4; /* 控制背景透明度 */
  z-index: -2;
}

/* 白色渐变边缘遮罩层 */
.login-page::after {
  content: "";
  position: absolute;
  top: 0; left: 0;
  width: 100vw;
  height: 100vh;
  background: radial-gradient(circle, rgba(255,255,255,0.5) 0%, rgba(255,255,255,0.8) 80%, rgba(255,255,255,1) 100%);
  z-index: -1;
}

</style>
  