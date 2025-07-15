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

const handleKeyDown = (e) => {
  if (e.shiftKey) {
    // Shift+Enter 正常换行
    return
  } else {
    // 只按 Enter 时，发送消息并阻止默认行为（防止回车换行）
    e.preventDefault()
    sendMessage()
  }
}

// 格式化文本，将换行符替换为 <br>
const formatText = (text) => {
  return text.replace(/\n/g, '<br>')
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (chatMessages.value) {
      chatMessages.value.scrollTop = chatMessages.value.scrollHeight
    }
  })
}

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
  scrollToBottom()
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
            <div>{{ msg.sender === 'user' ? '管理员' : 'DeepSeek 助手' }}</div>
          </div>
          <div class="message-bubble" v-html="formatText(msg.text)"></div>
        </div>
      </div>
      <div class="chat-input-container">
        <div class="chat-input-box">
          <textarea
            v-model="newMessage"
            class="chat-input"
            placeholder="输入消息..."
            @keydown.enter="handleKeyDown"
          />
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
  background: #f0f4f8; /* 浅灰背景 */
  overflow: hidden;   /* 防止内部撑出滚动条 */
}

.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #ffffff;
  border-bottom: 1px solid #e2e8f0;
  padding: 12px 20px;
}

.title {
  font-size: 18px;
  font-weight: 600;
  color: #1d4ed8;
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
  color: #1d4ed8;
  font-weight: 500;
}

.logout-button {
  background: #1d4ed8;
  color: #fff;
  border: none;
  padding: 4px 10px;
  border-radius: 6px;
  cursor: pointer;
}

.right-time {
  font-size: 12px;
  color: #334155;
}

.fixed-date, .fixed-time {
  font-weight: 500;
  margin-left: 4px;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f8fafc;
  overflow: hidden;   /* 防止内部撑出滚动条 */
}

.chat-header {
  background: white;
  border-bottom: 1px solid #e2e8f0;
  padding: 10px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chat-title {
  font-weight: 600;
  color: #1e293b;
}

.chat-status {
  background: #10b981;
  color: white;
  font-size: 16px;
  padding: 2px 8px;
  border-radius: 10px;
  display: flex;
  align-items: center;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 150px;
  display: flex;
  flex-direction: column;
}

.message {
  display: inline-flex;          /* 用 inline-flex 而不是 flex，内容决定宽度 */
  flex-direction: column;
  margin-bottom: 14px;
  max-width: 80%; 
}

.user-message {
  align-self: flex-end;
  text-align: right;
}

.ai-message {
  align-self: flex-start;
  text-align: left;
}

/* header 部分对齐方式调整 */
.user-message .message-header {
  flex-direction: row-reverse; /* icon 和用户名到右边 */
}

.ai-message .message-header {
  flex-direction: row; /* icon 和用户名到左边 */
}

.message-header {
  display: flex;
  align-items: center;
  font-size: 15px;
  margin-bottom: 4px;
  color: #64748b;
}

/* icon 本身不需要变动，只是跟随 flex-direction */
.message-icon {
  width: 22px;
  height: 22px;
  background: #1d4ed8;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  margin: 0 6px; /* 左右都留空隙 */
}

.user-message .message-bubble {
  background: #dbeafe; /* 浅蓝色 */
  color: #1e3a8a;
  border-radius: 12px 0 12px 12px;
  margin-right: 28px; 
}

.ai-message .message-bubble {
  background: #e2e8f0; /* 浅灰色 */
  color: #334155;
  border-radius: 0 12px 12px 12px;
  margin-left: 28px; 
}

.message-bubble {
  padding: 16px 20px;
  font-size: 14px;
  line-height: 1.5;
}

.chat-input-container {
  border-top: 1px solid #e2e8f0;
  background: white;
  padding: 10px 20px;
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
  font-size: 14px;
}

.send-button {
  background: #1d4ed8;
  border: none;
  color: white;
  margin-left: 8px;
  padding: 0 12px;
  border-radius: 8px;
  cursor: pointer;
}
</style>
