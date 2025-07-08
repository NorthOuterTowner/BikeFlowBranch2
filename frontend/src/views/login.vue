<template>
    <div class="login-page">
      <h1>登录</h1>
      <button @click="goDashboard">登录按钮</button>
    </div>
  </template>
  
  <script setup>
  import { useRouter } from 'vue-router'
  const router = useRouter()
  
  function goDashboard() {
    // 登录成功后跳转
    router.push('/dashboard')
  }
  </script>
  
  <style scoped>
  .login-page {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    color: aliceblue;
  }
  </style>
  