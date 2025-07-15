<script setup>
import { ref, computed, nextTick } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

// 欢迎信息 & 时间
const welcoming = ref('管理员，欢迎您！')
const fixedDate = computed(() => new Date().toLocaleDateString())
const currentHour = computed(() => new Date().getHours() + ':00')

// 对话内容
const newMessage = ref('')
const messages = ref([
  { sender: 'ai', text: '你好！我是DeepSeek智能助手。请问有什么可以帮您？' },
  { sender: 'user', text: '我正在开发一个接入DeepSeek的项目，有什么建议？' },
  { sender: 'ai', text: '建议：1. 简洁UI；2. 区分用户与AI消息；3. 添加API配置区域等。' }
])

// 聊天消息容器
const chatMessages = ref(null)

// 发送消息
const sendMessage = () => {
  const text = newMessage.value.trim()
  if (!text) return

  messages.value.push({ sender: 'user', text })
  newMessage.value = ''

  // 模拟AI回复
  setTimeout(() => {
    messages.value.push({ sender: 'ai', text: '这是模拟回复，实际应由API返回。' })
    nextTick(() => {
      if (chatMessages.value) {
        chatMessages.value.scrollTop = chatMessages.value.scrollHeight
      }
    })
  }, 800)
}

// 退出
const logout = async () => {
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    // 清除所有 sessionStorage 项
    sessionStorage.clear()
    router.push('/login')
  }
}
</script>


<template>
  <div class="app-container">
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车潮汐预测调度——AI助手</h1>
      </div>
      <div class="user-info"> 
        <div class="user-top">  
          <span class="welcoming">{{ welcoming }}</span> 
          <button class="logout-button" @click="logout">退出</button>
        </div>
        <div class="right-time">
          <label>日期：</label>
          <span class="fixed-date">{{ fixedDate }}</span>
          <label>当前时段：</label>
          <span class="fixed-time">{{ currentHour }}</span>
        </div>
      </div>
    </header>

    <!-- 对话面板 -->
    <div class="chat-container">
      <div class="chat-header">
        <div class="chat-title">DeepSeek 对话助手</div>
        <div class="chat-status">
          <i class="fas fa-circle"></i> 已连接
        </div>
      </div>
      <div class="chat-messages" ref="chatMessages">
        <div v-for="(msg, index) in messages" :key="index" :class="['message', msg.sender + '-message']">
          <div class="message-header">
            <div class="message-icon">
              <i :class="msg.sender === 'user' ? 'fas fa-user' : 'fas fa-robot'"></i>
            </div>
            <div>{{ msg.sender === 'user' ? '前端工程师' : 'DeepSeek 助手' }}</div>
          </div>
          <div class="message-bubble">{{ msg.text }}</div>
        </div>
      </div>
      <div class="chat-input-container">
        <div class="chat-input-box">
          <textarea v-model="newMessage" class="chat-input" placeholder="输入消息..." @keypress.enter.prevent="sendMessage"></textarea>
          <button class="send-button" @click="sendMessage">
            <i class="fas fa-paper-plane"></i>
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
  flex-shrink: 0;
}

.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
}

.user-info {
  text-align: right;
}

.user-top {
  display: flex;
  align-items: center;
  gap: 10px;
}
.welcoming {
  color: #091275;
  font-weight: bold;
}
.logout-button { background:#091275; color:white; border:none; padding:4px 8px; border-radius:4px; cursor:pointer; }
.right-time .fixed-date {
  margin-right: 20px;
  font-weight: bold;
  color: #091275;
}
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
}

.chat-header {
  padding: 10px 20px;
  background: white;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-title {
  font-weight: bold;
}

.chat-status {
  background: #10b981;
  color: white;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 12px;
  display: flex;
  align-items: center;
}

.chat-messages {
  flex: 1;
  padding: 10px 20px;
  overflow-y: auto;
}

.message {
  max-width: 70%;
  margin-bottom: 10px;
}

.user-message {
  align-self: flex-end;
}

.ai-message {
  align-self: flex-start;
}

.message-header {
  display: flex;
  align-items: center;
  font-size: 12px;
  margin-bottom: 4px;
}

.message-icon {
  width: 20px;
  height: 20px;
  background: #2563eb;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  margin-right: 6px;
}

.message-bubble {
  background: white;
  padding: 8px 12px;
  border-radius: 12px;
}

.chat-input-container {
  padding: 10px 20px;
  border-top: 1px solid #e2e8f0;
  background: white;
}

.chat-input-box {
  display: flex;
}

.chat-input {
  flex: 1;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  padding: 8px;
  resize: none;
  outline: none;
}

.send-button {
  margin-left: 8px;
  background: #2563eb;
  border: none;
  color: white;
  padding: 0 12px;
  border-radius: 8px;
  cursor: pointer;
}
</style>
