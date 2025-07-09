<template>
  <div class="login-page">
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
  import { login } from '../api/auth' // 假设你有一个 auth.js 文件处理登录请求
  import { useMessageStore } from '../store/messageStore'  // 根据实际路径调整
  import { NInput, NButton, NAlert, NSpin } from 'naive-ui'

  const account = ref('')
  const password = ref('')
  const router = useRouter()
  const messageStore = useMessageStore()
  const loading = ref(false)
  
  async function doLogin() {
    messageStore.setMessage('', '') // 清空提示
    try {
      const res = await login(account.value, password.value)
      if (res.data.code === 200) {
        messageStore.setMessage('登录成功', 'success')
        // 存储 token 等信息
        localStorage.setItem('token', res.data.data.token)
        localStorage.setItem('username', res.data.data.account)

        loading.value = true // 开始加载状态
        // 等 1.5 秒后再跳转，让用户看到提示
        setTimeout(() => {
          loading.value = false // 结束加载状态
          router.push('/dashboard')
        }, 1500)
      } else {
        messageStore.setMessage('登录失败: ' + res.data.msg, 'error')
      }
    } catch (e) {
      console.error(e)
      messageStore.setMessage('请求失败，请检查网络或后端服务', 'error')
    }
  }
</script>
  
<style scoped>
.login-page {
  display: flex;
  flex-direction: column;
  align-items: center;      /* 水平居中 */
  justify-content: center;  /* 垂直居中 */
  height: 100vh;
  max-width: 300px;
  width: 90%;              /* 小屏幕时也好看 */
  margin: 0 auto;
  text-align: center;
  padding: 0 16px;  
  /* background: linear-gradient(135deg, #ffffff, #0556a7); */
}
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

</style>
  