<script setup>
import { ref, computed, nextTick, watch, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { postSuggestion, postDispatchPlan } from '../../api/axios'
import request from '@/api/axios'
import { ElMessage } from 'element-plus'

const router = useRouter()

// 欢迎信息 & 时间
const welcoming = ref('管理员，欢迎您！')
const fixedDate = localStorage.getItem('selectedDate'); // "2025-07-17"
const currentHour = localStorage.getItem('selectedHour'); // "14:00"

const target_time = getTargetTime(fixedDate, currentHour);
console.log('target_time:', target_time);  // 输出类似：2025-07-17T06:00:00.000Z
function getTargetTime(fixedDate, currentHour) {
  // 如果没传，默认取当前时间
  if (!fixedDate) {
    fixedDate = new Date().toISOString().split('T')[0]; // "2025-07-17"
  }

  if (!currentHour) {
    currentHour = new Date().getHours() + ':00';
  }

  const mapHourToSegment = (hour) => {
    if (hour < 3) return 0;
    if (hour < 6) return 3;
    if (hour < 9) return 6;
    if (hour < 12) return 9;
    if (hour < 15) return 12;
    if (hour < 18) return 15;
    if (hour < 21) return 18;
    return 21;
  };

  const hour = parseInt(currentHour.split(':')[0]);
  const mappedHour = mapHourToSegment(hour);

  const localDateTime = new Date(`${fixedDate}T${String(mappedHour).padStart(2, '0')}:00:00`);
  const targetTime = new Date(localDateTime.getTime() - localDateTime.getTimezoneOffset() * 60000).toISOString();

  return targetTime;
}

// 当前模式：'chat' 或 'plan'
const currentMode = ref('chat')

// 聊天消息容器 DOM
const chatMessagesEl = ref(null)

// 两个对话列表
const chatMessages = ref([])
const planMessages = ref([])

// 是否已发送初始化消息
const hasSentChatInit = ref(false)
const hasSentPlanInit = ref(false)

// 新消息输入
const newMessage = ref('')

// 优化方案相关
const selectedPlanIds = ref([])
const planLoading = ref(false)

// 当前显示的消息列表
const messages = computed(() => {
  return currentMode.value === 'chat' ? chatMessages.value : planMessages.value
})

// 滚动到底部
const scrollToBottom = () => {
  nextTick(() => {
    if (chatMessagesEl.value) {
      chatMessagesEl.value.scrollTop = chatMessagesEl.value.scrollHeight
    }
  })
}

// 格式化文本：换行符 -> <br>
const formatText = (text) => (text ?? '').replace(/\n/g, '<br>')

// ---------------------------
// ✅ 本地缓存：加载

onMounted(() => {
  const token = sessionStorage.getItem('token')
  if (!token) {
    console.warn('未找到 token，不加载缓存')
    return
  }

  const savedChat = localStorage.getItem(`chatMessages_${token}`)
  const savedPlan = localStorage.getItem(`planMessages_${token}`)

  if (savedChat) {
    chatMessages.value = JSON.parse(savedChat)
    hasSentChatInit.value = true
  } else {
    hasSentChatInit.value = false
  }

  if (savedPlan) {
    planMessages.value = JSON.parse(savedPlan)
    hasSentPlanInit.value = true
  } else {
    hasSentPlanInit.value = false
  }

  sendInitMessage(currentMode.value)
})

// ✅ 本地缓存：监听写入
watch(chatMessages, (newVal) => {
  const token = sessionStorage.getItem('token')
  if (token) {
    localStorage.setItem(`chatMessages_${token}`, JSON.stringify(newVal))
  }
}, { deep: true })

watch(planMessages, (newVal) => {
  const token = sessionStorage.getItem('token')
  if (token) {
    localStorage.setItem(`planMessages_${token}`, JSON.stringify(newVal))
  }
}, { deep: true })

// ---------------------------
// 发送消息（普通聊天）
async function sendMessage() {
  const text = newMessage.value.trim()
  if (!text) return

  chatMessages.value.push({ sender: 'user', text, loading: false })
  newMessage.value = ''

  const loadingMsg = { sender: 'ai', text: '正在深度思考中，请稍候...', loading: true }
  chatMessages.value.push(loadingMsg)

  try {
    const reply = await postSuggestion(target_time, text)
    const idx = chatMessages.value.indexOf(loadingMsg)
    if (idx !== -1) {
      chatMessages.value[idx] = { sender: 'ai', text: reply }
    }
  } catch (e) {
    console.error('请求失败', e)
    const idx = chatMessages.value.indexOf(loadingMsg)
    if (idx !== -1) {
      chatMessages.value[idx] = { sender: 'ai', text: '请求失败，请稍后再试' }
    }
    ElMessage.error('请求失败')
  }
  scrollToBottom()
}

// 发送优化方案
async function generateOptimizedPlan() {
  const text = newMessage.value.trim()
  if (!text) return

  planMessages.value.push({ sender: 'user', text, loading: false })
  newMessage.value = ''

  const loadingMsg = { sender: 'ai', text: '正在生成优化方案...', loading: true }
  planMessages.value.push(loadingMsg)
  planLoading.value = true

  try {
    const res = await postDispatchPlan(target_time, text)
    const idx = planMessages.value.indexOf(loadingMsg)
    if (idx !== -1) {
      planMessages.value.splice(idx, 1, {
        sender: 'ai',
        text: `已生成 ${res.optimized_plan.length} 条调度建议：`,
        data: res.optimized_plan,
        selectable: true,     // ✅ 新增：表示可选
        loading: false
      })
    }
  } catch (e) {
    console.error('生成失败', e)
    const idx = planMessages.value.indexOf(loadingMsg)
    if (idx !== -1) {
      planMessages.value.splice(idx, 1, {
        sender: 'ai',
        text: e.message || '生成失败，请稍后重试',
        loading: false
      })
    }
    ElMessage.error('生成失败')
  } finally {
    planLoading.value = false
    scrollToBottom()
  }
}


// 采纳方案
async function acceptSelectedPlans() {
  if (selectedPlanIds.value.length === 0) {
    ElMessage.warning('请先选择要采纳的方案')
    return
  }

  // 始终找到最新一条可采纳的消息
  const lastPlanMsg = [...planMessages.value].reverse().find(
    m => m.sender === 'ai' && m.data && m.selectable
  )
  if (!lastPlanMsg) {
    ElMessage.error('未找到可采纳的方案')
    return
  }

  const acceptedItems = [] // 已采纳成功
  const failedItems = []   // 采纳失败

  // 根据当前索引找到要采纳的项
  const itemsToAccept = selectedPlanIds.value.map(idx => lastPlanMsg.data[idx])

  for (const item of itemsToAccept) {
    try {
      await request.post('/dispatch/add', {
        schedule_time: target_time,
        start_station_id: item.from_station_id,
        end_station_id: item.to_station_id,
        bikes_to_move: item.bikes_to_move,
        remark: '由AI生成'
      })
      acceptedItems.push(item)
      ElMessage.success(`采纳成功：${item.from_station_id} → ${item.to_station_id}`)
    } catch (e) {
      console.error('采纳失败', e)
      failedItems.push(item)
      ElMessage.error('采纳失败，请稍后再试')
    }
  }

  // ✅ 从原消息里移除已采纳项
  lastPlanMsg.data = lastPlanMsg.data.filter(item => !acceptedItems.includes(item))

  // ✅ 如果采纳了，新增一条已采纳消息（只展示，不可选）
  if (acceptedItems.length > 0) {
    planMessages.value.push({
      sender: 'ai',
      text: `已采纳 ${acceptedItems.length} 条调度方案：`,
      data: acceptedItems,
      selectable: false
    })
  }

  // ✅ 清空勾选
  selectedPlanIds.value = []
  scrollToBottom()
}


// 初始化提示
function sendInitMessage(mode) {
  if (mode === 'chat') {
    if (!hasSentChatInit.value && chatMessages.value.length === 0) {
      chatMessages.value.push({
        sender: 'ai',
        text: '这是普通聊天模式，随便跟我说说什么吧。'
      })
      hasSentChatInit.value = true
    }
  } else if (mode === 'plan') {
    if (!hasSentPlanInit.value && planMessages.value.length === 0) {
      planMessages.value.push({
        sender: 'ai',
        text: '请输入需求，我会帮你生成调度方案。'
      })
      hasSentPlanInit.value = true
    }
  }
}

// 切换模式时发送初始化提示
watch(currentMode, (mode) => {
  sendInitMessage(mode)
  scrollToBottom()
})

// 回车处理
const handleKeyDown = (e) => {
  if (e.shiftKey) return
  e.preventDefault()
  if (currentMode.value === 'chat') {
    sendMessage()
    scrollToBottom()
  } else {
    generateOptimizedPlan()
    scrollToBottom()
  }
}

// 退出
const logout = async () => {
  try {
    await request.post('/api/user/logout')
  } catch (error) {
    console.warn('登出失败，可忽略', error)
  } finally {
    const token = sessionStorage.getItem('token')
    if (token) {
      localStorage.removeItem(`chatMessages_${token}`)
      localStorage.removeItem(`planMessages_${token}`)
    }
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
            <div class="message-icon" v-if="msg.sender === 'ai'">
              <img
                src="/icons/ds.png"
                alt="AI头像"
                style="width: 100%; height: 100%; object-fit: cover; border-radius: 50%;"
              />
            </div>
            <div class="message-name">
              {{ msg.sender === 'user' ? '管理员' : 'DeepSeek 助手' }}
              <span v-if="msg.loading" class="spinner-small"></span>
            </div>
          </div>

          <div class="message-bubble" v-html="formatText(msg.text)"></div>

          <!-- 如果 msg 有 data，就在这里显示交互式方案列表 -->
          <div v-if="msg.data" class="message-bubble" style="background:#fff; margin-top:8px;">
            <ul style="list-style: none; padding: 0; margin:0;">
              <li v-for="(item, i) in msg.data" :key="i" style="margin-bottom: 8px;">
                <template v-if="msg.selectable">
                  <!-- 可勾选 -->
                  <label style="cursor: pointer;">
                    <input 
                      type="checkbox"
                      v-model="selectedPlanIds"
                      :value="i"
                      style="margin-right: 6px;"
                    />
                    从 <strong>{{ item.from_station_id }}</strong> → <strong>{{ item.to_station_id }}</strong>，
                    调度 <strong>{{ item.bikes_to_move }}</strong> 辆
                  </label>
                </template>
                <template v-else>
                  <!-- 已采纳，只展示 -->
                  从 <strong>{{ item.from_station_id }}</strong> → <strong>{{ item.to_station_id }}</strong>，
                  调度 <strong>{{ item.bikes_to_move }}</strong> 辆
                </template>
                <div style="color: gray; font-size: 12px; margin-left: 24px;">
                  {{ item.reason }}
                </div>
              </li>
            </ul>

            <!-- 只有 selectable 才显示采纳按钮 -->
            <button 
              v-if="msg.selectable"
              class="send-button" 
              @click="acceptSelectedPlans"
            >
              采纳选中方案
            </button>
          </div>



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
