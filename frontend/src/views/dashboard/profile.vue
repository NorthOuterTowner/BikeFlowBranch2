<template>
  <header class="header">
    <div class="user-info">
      <div class="user-top">
        <span class="welcoming">{{ welcoming }}</span>
        <button class="logout-button" @click="logout">退出</button>
      </div>
    </div>
  </header>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const welcoming = ref('管理员，欢迎您！')
const loading = ref(false)
const dateTime = ref(new Date().toISOString().slice(0, 16)) // yyyy-MM-ddTHH:mm

const logout = async () => {
  try {
    await fetch('/api/user/logout', { method: 'POST' })
  } catch (err) {
    console.warn('退出失败', err)
  }
  router.push('/login')
}

const refreshData = async () => {
  loading.value = true
  setTimeout(() => {
    loading.value = false
    alert('数据刷新成功')
  }, 1000)
}
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

.welcoming {
  font-size: 14px;
  white-space: nowrap; 
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


.logout-button{
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

</style>
