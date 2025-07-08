<template>
    <div class="login-page">
        <h1>登录</h1>
        <input v-model="account" placeholder="用户名" />
        <input v-model="password" type="password" placeholder="密码" />
        <button @click="doLogin">登录</button>

        <p class="to-register">
            没有账号？
            <router-link to="/register">去注册</router-link>
        </p>
    </div>
  </template>
  
<script setup>
  import { ref } from 'vue'
  import { useRouter } from 'vue-router'
  import { login } from '../api/auth' // 假设你有一个 auth.js 文件处理登录请求

  const account = ref('')
  const password = ref('')
  const router = useRouter()
  
  async function doLogin() {
  try {
    const res = await login(account.value, password.value)
    // const account = 'admin';     // 写死的用户名
    // const password = 'admin';        // 写死的密码
    // console.log('提交的用户名:', username);
    // console.log('提交的密码:', password);

    // const res = await login(account, password);
    // console.log('后端返回:', res);

    if (res.data.code === 200) {
      alert('登录成功')
      // 存储 token 等信息
      localStorage.setItem('token', res.data.data.token)
      localStorage.setItem('username', res.data.data.account)
      // 跳转到 dashboard
      router.push('/dashboard')
    } else {
      alert('登录失败: ' + res.data.msg)
    }
  } catch (e) {
    console.error(e)
    alert('请求失败，请检查网络或后端服务')
  }
}
</script>
  
<style scoped>
  .login-page {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    background: #f0f2f5;
}
input {
  margin: 8px 0;
  padding: 8px;
  width: 200px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
button {
  padding: 8px 16px;
  border: none;
  background-color: #409eff;
  color: white;
  border-radius: 4px;
  cursor: pointer;
}
.to-register {
  margin-top: 16px;
}
button:hover {
  background-color: #66b1ff;
}
</style>
  