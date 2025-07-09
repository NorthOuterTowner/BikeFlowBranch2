<template>
  <header class="header">
    <div class="user-info">
      <div class="user-top">
        <span class="welcoming">{{ welcoming }}</span>
        <button class="logout-button" @click="logout">退出</button>
      </div>
    </div>
  </header>

  <div class="profile-container">
    <table class="profile-table">
      <tr>
        <th colspan="2" class="profile-title">个人信息</th>
      </tr>
      <tr>
        <td class="label">账号</td>
        <td>{{ account }}</td>
      </tr>
      <tr>
        <td class="label">邮箱</td>
        <td>{{ email }}</td>
      </tr>
    </table>
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
.header {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 10px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ccc;
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

.welcoming {
  font-size: 14px;
  white-space: nowrap; 
}

.logout-button {
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
.logout-button:hover {
  background-color: #0d1c9e;
}

.profile-container {
  max-width: 500px;
  margin: 30px auto;
  padding: 20px;
}

.profile-table {
  width: 100%;
  height:100%;
  border-collapse: collapse;
  background-color: #ffffff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.profile-title {
  background-color: #091275;
  color: white;
  font-size: 18px;
  padding: 12px;
  text-align: left;
}

.profile-table td,
.profile-table th {
  padding: 12px;
  border-bottom: 1px solid #eee;
}

.label {
  background-color: #f5f5f5;
  width: 120px;
  font-weight: bold;
}
</style>
