<template>
    <div class="register-page">
      <h1>注册</h1>
      <input v-model="account" placeholder="用户名" />
      <input v-model="password" type="password" placeholder="密码" />
      <input v-model="email" placeholder="邮箱" />
      <button @click="doRegister">注册</button>
  
      <p class="to-login">
        已有账号？
        <router-link to="/login">去登录</router-link>
      </p>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import { useRouter } from 'vue-router'
  import { register } from '../api/auth'
  
  const account = ref('')
  const password = ref('')
  const email = ref('')
  const router = useRouter()
  
  async function doRegister() {
    try {
      const res = await register(account.value, password.value, email.value);
    //   const account = 'admin';     // 写死的用户名
    // const password = 'admin';        // 写死的密码
    // const email = '2499814010@qq.com';
    // console.log('提交的用户名:', account);
    // console.log('提交的密码:', password);

    // const res = await register(account, password,email);
    // console.log('后端返回:', res);

      if (res.data.code === 200) {
        alert('注册成功，请登录')
        router.push('/login')
      } else {
        alert('注册失败: ' + res.data.msg)
      }
    } catch (e) {
      console.error(e)
      alert('请求失败，请检查网络或后端服务')
    }
  }
  </script>
  
  <style scoped>
.register-page {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  height: 100vh;
  background: #f0f2f5;
}

input {
  padding: 8px;
  width: 200px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

button {
  padding: 8px 16px;
  border: none;
  background-color: #67c23a;
  color: white;
  border-radius: 4px;
  cursor: pointer;
}

button:hover {
  background-color: #85ce61;
}
</style>