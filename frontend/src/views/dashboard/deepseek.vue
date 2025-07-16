<script setup>
import { ref, computed, nextTick, watch, onMounted} from 'vue'
import { useRouter } from 'vue-router'
import { postSuggestion, postDispatchPlan } from '../../api/axios'
import request from '@/api/axios'

const router = useRouter()

// 欢迎信息 & 时间
const welcoming = ref('管理员，欢迎您！')
const fixedDate = computed(() => new Date().toLocaleDateString())
const currentHour = computed(() => new Date().getHours() + ':00')
const target_time = new Date('2025-06-13T09:35:00Z').toISOString();
  // 直接新建一个 ISO 时间字符串

// 当前模式：'chat' 或 'plan'
const currentMode = ref('chat')

// 聊天消息容器 DOM 元素
const chatMessagesEl = ref(null)

const isLoading = ref(false)
//const answer = ref(null)

// 两组对话
const chatMessages = ref([])
const planMessages = ref([])

//记录是否发送过初始化消息
const hasSentChatInit = ref(false)
const hasSentPlanInit = ref(false)

// 新消息输入框
const newMessage = ref('')

// 计算属性，根据当前模式返回对应的消息列表
const messages = computed(() => {
  return currentMode.value === 'chat' ? chatMessages.value : planMessages.value
})

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
  return (text ?? '').replace(/\n/g, '<br>')
}

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (chatMessagesEl.value) {
      chatMessagesEl.value.scrollTop = chatMessagesEl.value.scrollHeight
    }
  })
}

async function sendMessage() {
  const text = newMessage.value.trim()
  if (!text) return

  const msg = {
    sender: 'user',
    text
  }

  // 选择对应消息列表
  const messagesList = currentMode.value === 'chat' ? chatMessages.value : planMessages.value

  messagesList.push(msg)
  newMessage.value = ''

  // 加入 loading 占位消息
  const loadingMsg = { sender: 'ai', text: '正在深度思考中，请稍候...', loading: true }
  messagesList.push(loadingMsg)

  try {
    let reply
    if (currentMode.value === 'chat') {
      reply = await postSuggestion(target_time, text)
    } else {
      reply = await postDispatchPlan(target_time, text)
    }
    // 找到 loadingMsg 替换为真正回复
    const idx = messagesList.indexOf(loadingMsg)
    if (idx !== -1) {
      messagesList[idx] = { sender: 'ai', text: reply }
    }
  } catch (error) {
    const errMsg = '出错了，请稍后再试'
    const idx = messagesList.indexOf(loadingMsg)
    if (idx !== -1) {
      messagesList[idx] = { sender: 'ai', text: errMsg }
    }
  }

  nextTick(() => {
    if (chatMessagesEl.value) {
      chatMessagesEl.value.scrollTop = chatMessagesEl.value.scrollHeight
    }
    scrollToBottom()
  })
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

function sendInitMessage(mode) {
  if (mode === 'chat' && !hasSentChatInit.value) {
    chatMessages.value.push({
      sender: 'ai',
      text: '这是普通聊天模式，随便跟我说说什么吧。'
    })
    hasSentChatInit.value = true
  } else if (mode === 'plan' && !hasSentPlanInit.value) {
    planMessages.value.push({
      sender: 'ai',
      text: '在这里输入你想要的调度方案的修改，我会给你直接生成新的调度方案，你可以选择采纳或者不采纳。'
    })
    hasSentPlanInit.value = true
  }
}

watch(currentMode, (newMode) => {
  sendInitMessage(newMode)
})

onMounted(() => {
  sendInitMessage(currentMode.value)
})

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
        <div class="mode-switch">
          <button
            :class="{ active: currentMode === 'chat' }"
            @click="currentMode = 'chat'"
          >
            对话聊天
          </button>
          <button
            :class="{ active: currentMode === 'plan' }"
            @click="currentMode = 'plan'"
          >
            生成方案
          </button>
        </div>
        <div class="chat-status">
          <i class="fas fa-circle"></i> 已连接
        </div>
      </div>

      <div class="chat-messages" ref="chatMessagesEl">
        <div
          v-for="(msg, index) in messages"
          :key="index"
          :class="['message', msg.sender + '-message']"
        >
          <div class="message-header">
            <div class="message-icon">
              <i :class="msg.sender === 'user' ? 'fas fa-user' : 'fas fa-robot'"></i>
            </div>
            <div class="message-name">
              {{ msg.sender === 'user' ? '管理员' : 'DeepSeek 助手' }}
              <span v-if="msg.loading" class="spinner-small"></span>
            </div>
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
          ></textarea>
          <button class="send-button" @click="sendMessage">
            <i class="fas fa-paper-plane"></i>
          </button>
        </div>
      </div>

      <!-- 加载弹窗
      <div v-if="isLoading" class="modal-loading">
        <div class="spinner"></div>
        <p>正在深度思考中，请稍候...</p>
      </div> -->

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

.mode-switch {
  display: flex;
  border: 1px solid #007bff;
  border-radius: 6px;
  overflow: hidden;
}

.mode-switch button {
  padding: 4px 12px;
  font-size: 12px;
  background: transparent;
  color: #007bff;
  border: none;
  cursor: pointer;
  transition: background 0.3s;
}

.mode-switch button.active {
  background: #007bff;
  color: #fff;
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
  text-align: left;
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
  word-wrap: break-word;
  word-break: break-word;
  white-space: normal;
}

.chat-input-container {
  border-top: 1px solid #e2e8f0;
  background: rgb(255, 255, 255);
  padding: 20px 150px;
}

.chat-input-box {
  display: flex;
}

.chat-input {
  flex: 1;
  border: 1px solid #e2e8f0;
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
.modal-loading {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: white;
  font-size: 20px;
  z-index: 9999;
}
.spinner {
  border: 5px solid rgba(255,255,255,0.3);
  border-top-color: white;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin-bottom: 16px;
}
.spinner-small {
  display: inline-block;
  margin-left: 6px;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(0,0,0,0.2);
  border-top-color: #1d4ed8;  /* 深蓝 */
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
  vertical-align: middle;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.message-name {
  display: flex;
  align-items: center;
}

</style>
