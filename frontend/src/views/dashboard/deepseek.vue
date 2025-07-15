<template>
    <div class="container">
      <!-- 左侧导航 -->
      <div class="sidebar">
        <div class="logo">
          <div class="logo-icon">
            <i class="fas fa-brain"></i>
          </div>
          <div class="logo-text">DeepSeek</div>
        </div>
        <div class="nav-links">
          <div class="nav-item active">
            <div class="nav-icon"><i class="fas fa-comments"></i></div>
            <div class="nav-text">对话界面</div>
          </div>
          <div class="nav-item">
            <div class="nav-icon"><i class="fas fa-code"></i></div>
            <div class="nav-text">API 接入</div>
          </div>
          <div class="nav-item">
            <div class="nav-icon"><i class="fas fa-sliders-h"></i></div>
            <div class="nav-text">模型设置</div>
          </div>
          <div class="nav-item">
            <div class="nav-icon"><i class="fas fa-history"></i></div>
            <div class="nav-text">历史记录</div>
          </div>
          <div class="nav-item">
            <div class="nav-icon"><i class="fas fa-cog"></i></div>
            <div class="nav-text">系统设置</div>
          </div>
        </div>
        <div class="footer">
          <p>DeepSeek API v1.2.5</p>
          <p>© 2023 DeepSeek Technologies</p>
        </div>
      </div>
  
      <!-- 中间聊天区 -->
      <div class="chat-container">
        <div class="chat-header">
          <div class="chat-title">DeepSeek 对话助手</div>
          <div class="chat-status">
            <i class="fas fa-circle"></i> 已连接
          </div>
        </div>
        <div class="chat-messages" ref="chatMessages">
          <div
            v-for="(msg, index) in messages"
            :key="index"
            :class="['message', msg.sender === 'user' ? 'user-message' : 'ai-message']"
          >
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
            <textarea
              class="chat-input"
              v-model="inputText"
              placeholder="输入消息..."
              @keypress.enter.prevent="handleSend"
            ></textarea>
            <button class="send-button" @click="handleSend">
              <i class="fas fa-paper-plane"></i>
            </button>
          </div>
        </div>
      </div>
  
      <!-- 右侧配置面板 -->
      <div class="config-panel">
        <div class="panel-title">
          <i class="fas fa-cogs"></i> API 配置
        </div>
        <div class="form-group">
          <label class="form-label">API 密钥</label>
          <input type="password" class="form-control" placeholder="输入您的API密钥" />
        </div>
        <div class="form-group">
          <label class="form-label">模型选择</label>
          <select class="form-control">
            <option>DeepSeek-Vision (多模态)</option>
            <option selected>DeepSeek-Coder (代码专用)</option>
            <option>DeepSeek-Chat (通用对话)</option>
          </select>
        </div>
        <div class="form-group">
          <label class="form-label">温度参数 (0-1)</label>
          <input type="range" class="form-control" min="0" max="1" step="0.1" v-model="temperature" />
          <div style="text-align: center; margin-top: 5px; font-size: 13px;">{{ temperature }}</div>
        </div>
        <div class="form-group">
          <label class="form-label">最大生成长度</label>
          <input type="number" class="form-control" v-model="maxTokens" />
        </div>
        <div class="checkbox-group">
          <input type="checkbox" v-model="useStream" />
          <label>启用流式响应</label>
        </div>
        <div class="checkbox-group">
          <input type="checkbox" v-model="keepHistory" />
          <label>保留对话历史</label>
        </div>
        <div class="divider"></div>
        <div class="form-group">
          <label class="form-label">API 端点</label>
          <input type="text" class="form-control" v-model="apiEndpoint" />
        </div>
        <button class="btn btn-block" style="margin-top: 20px;">
          <i class="fas fa-plug"></i> 连接 API
        </button>
        <div class="divider"></div>
        <div class="panel-title">
          <i class="fas fa-code"></i> 响应预览
        </div>
        <div class="response-box">{{ apiResponse }}</div>
        <div style="display: flex; gap: 10px; margin-top: 15px;">
          <button class="btn btn-outline"><i class="fas fa-copy"></i> 复制响应</button>
          <button class="btn btn-outline"><i class="fas fa-redo"></i> 重新生成</button>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, nextTick } from 'vue'
  
  const inputText = ref('')
  const messages = ref([
    { sender: 'ai', text: '你好！我是DeepSeek智能助手。请问有什么可以帮您的？' }
  ])
  
  const temperature = ref(0.7)
  const maxTokens = ref(1024)
  const useStream = ref(true)
  const keepHistory = ref(true)
  const apiEndpoint = ref('https://api.deepseek.com/v1/chat/completions')
  const apiResponse = ref(`{
    "id": "chatcmpl-8eJ7...",
    "object": "chat.completion",
    "created": 1697824180,
    "model": "DeepSeek-Coder",
    "choices": [
      { "index": 0, "message": { "role": "assistant", "content": "以下是前端接入界面的设计建议..." }, "finish_reason": "stop" }
    ],
    "usage": { "prompt_tokens": 56, "completion_tokens": 342, "total_tokens": 398 }
  }`)
  
  const chatMessages = ref(null)
  
  function handleSend() {
    const text = inputText.value.trim()
    if (!text) return
    messages.value.push({ sender: 'user', text })
    inputText.value = ''
    // 模拟 AI 响应
    setTimeout(() => {
      messages.value.push({
        sender: 'ai',
        text: '感谢您的消息！这是一个模拟响应。实际应替换为调用API后的返回结果。'
      })
      nextTick(() => {
        chatMessages.value.scrollTop = chatMessages.value.scrollHeight
      })
    }, 800)
  }
  </script>
  
  <style scoped>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    :root {
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --secondary: #64748b;
      --light: #f8fafc;
      --dark: #0f172a;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --user-bubble: #dbeafe;
      --ai-bubble: #e2e8f0;
      --sidebar-bg: #0f172a;
      --config-bg: #f1f5f9;
    }
    
    body {
      background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
      color: var(--dark);
      min-height: 100vh;
      display: flex;
      padding: 20px;
    }
    
    #app {
      display: flex;
      max-width: 1600px;
      width: 100%;
      margin: 0 auto;
      background: white;
      border-radius: 16px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
      overflow: hidden;
      height: calc(100vh - 40px);
    }
    
    /* 左侧导航 */
    .sidebar {
      width: 260px;
      background: var(--sidebar-bg);
      color: white;
      padding: 25px 0;
      display: flex;
      flex-direction: column;
      transition: all 0.3s;
    }
    
    .logo {
      display: flex;
      align-items: center;
      padding: 0 25px 30px;
      border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .logo-icon {
      font-size: 32px;
      color: var(--primary);
      margin-right: 12px;
    }
    
    .logo-text {
      font-size: 24px;
      font-weight: 700;
    }
    
    .nav-links {
      padding: 30px 0;
      flex: 1;
    }
    
    .nav-item {
      padding: 15px 25px;
      display: flex;
      align-items: center;
      cursor: pointer;
      transition: all 0.3s;
    }
    
    .nav-item:hover {
      background: rgba(255,255,255,0.1);
    }
    
    .nav-item.active {
      background: var(--primary);
    }
    
    .nav-icon {
      font-size: 20px;
      margin-right: 15px;
      width: 24px;
      text-align: center;
    }
    
    .nav-text {
      font-size: 16px;
      font-weight: 500;
    }
    
    .footer {
      padding: 20px 25px 0;
      border-top: 1px solid rgba(255,255,255,0.1);
      color: rgba(255,255,255,0.7);
      font-size: 14px;
    }
    
    /* 中间聊天区 */
    .chat-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      border-right: 1px solid #e2e8f0;
    }
    
    .chat-header {
      padding: 20px 30px;
      border-bottom: 1px solid #e2e8f0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    
    .chat-title {
      font-size: 20px;
      font-weight: 600;
      color: var(--dark);
    }
    
    .chat-actions {
      display: flex;
      gap: 10px;
    }
    
    .action-btn {
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: var(--light);
      color: var(--dark);
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .action-btn:hover {
      background: var(--primary);
      color: white;
    }
    
    .chat-status {
      margin-left: 15px;
      font-size: 14px;
      padding: 4px 12px;
      background: var(--success);
      color: white;
      border-radius: 20px;
      display: flex;
      align-items: center;
    }
    
    .chat-status i {
      margin-right: 5px;
      font-size: 10px;
    }
    
    .chat-messages {
      flex: 1;
      padding: 25px 30px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }
    
    .message {
      max-width: 80%;
      margin-bottom: 20px;
      opacity: 0;
      transform: translateY(10px);
      animation: fadeIn 0.3s ease forwards;
    }
    
    @keyframes fadeIn {
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .user-message {
      align-self: flex-end;
    }
    
    .ai-message {
      align-self: flex-start;
    }
    
    .message-bubble {
      padding: 15px 20px;
      border-radius: 18px;
      line-height: 1.5;
      position: relative;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .user-message .message-bubble {
      background: var(--user-bubble);
      border-bottom-right-radius: 5px;
      color: var(--dark);
    }
    
    .ai-message .message-bubble {
      background: var(--ai-bubble);
      border-bottom-left-radius: 5px;
    }
    
    .message-header {
      display: flex;
      align-items: center;
      margin-bottom: 8px;
      font-weight: 600;
      font-size: 14px;
    }
    
    .user-message .message-header {
      color: var(--primary-dark);
    }
    
    .ai-message .message-header {
      color: var(--secondary);
    }
    
    .message-icon {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-right: 10px;
      font-size: 14px;
    }
    
    .user-message .message-icon {
      background: var(--primary);
      color: white;
    }
    
    .ai-message .message-icon {
      background: var(--secondary);
      color: white;
    }
    
    .chat-input-container {
      padding: 20px 30px;
      border-top: 1px solid #e2e8f0;
    }
    
    .chat-input-box {
      display: flex;
      background: var(--light);
      border-radius: 12px;
      padding: 8px;
      transition: all 0.3s;
    }
    
    .chat-input-box:focus-within {
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.2);
    }
    
    .chat-input {
      flex: 1;
      border: none;
      padding: 12px 15px;
      background: transparent;
      resize: none;
      height: 50px;
      font-size: 16px;
      outline: none;
      color: var(--dark);
    }
    
    .send-button {
      width: 50px;
      height: 50px;
      border-radius: 10px;
      background: var(--primary);
      color: white;
      border: none;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      transition: all 0.2s;
    }
    
    .send-button:hover {
      background: var(--primary-dark);
    }
    
    .send-button i {
      font-size: 20px;
    }
    
    /* 右侧配置面板 */
    .config-panel {
      width: 380px;
      padding: 25px;
      background: var(--config-bg);
      overflow-y: auto;
      transition: all 0.3s;
    }
    
    .panel-title {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 25px;
      color: var(--dark);
      display: flex;
      align-items: center;
    }
    
    .panel-title i {
      margin-right: 10px;
      color: var(--primary);
    }
    
    .form-group {
      margin-bottom: 20px;
    }
    
    .form-label {
      display: block;
      margin-bottom: 8px;
      font-weight: 500;
      color: var(--dark);
      font-size: 14px;
    }
    
    .form-control {
      width: 100%;
      padding: 12px 15px;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      background: white;
      font-size: 14px;
      transition: all 0.2s;
    }
    
    .form-control:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }
    
    .checkbox-group {
      display: flex;
      align-items: center;
      margin: 15px 0;
    }
    
    .checkbox-group input {
      margin-right: 10px;
      width: 18px;
      height: 18px;
    }
    
    .divider {
      height: 1px;
      background: #cbd5e1;
      margin: 25px 0;
    }
    
    .btn {
      display: inline-block;
      padding: 12px 25px;
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
      text-align: center;
    }
    
    .btn:hover {
      background: var(--primary-dark);
    }
    
    .btn-block {
      display: block;
      width: 100%;
    }
    
    .btn-outline {
      background: transparent;
      border: 1px solid var(--primary);
      color: var(--primary);
    }
    
    .btn-outline:hover {
      background: var(--primary);
      color: white;
    }
    
    .response-box {
      background: white;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      padding: 15px;
      margin-top: 15px;
      font-size: 14px;
      font-family: monospace;
      max-height: 200px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
    
    /* 响应式设计 */
    @media (max-width: 1200px) {
      #app {
        flex-direction: column;
        height: auto;
      }
      
      .sidebar {
        width: 100%;
        padding: 15px;
      }
      
      .nav-links {
        display: flex;
        padding: 15px 0;
        overflow-x: auto;
      }
      
      .nav-item {
        padding: 10px 15px;
        white-space: nowrap;
      }
      
      .config-panel {
        width: 100%;
      }
      
      .chat-container {
        border-right: none;
        border-bottom: 1px solid #e2e8f0;
      }
    }
    
    @media (max-width: 768px) {
      body {
        padding: 10px;
      }
      
      .chat-header {
        padding: 15px;
      }
      
      .chat-messages {
        padding: 15px;
      }
      
      .message {
        max-width: 90%;
      }
      
      .chat-input-container {
        padding: 15px;
      }
    }
    
    /* 加载动画 */
    .typing-indicator {
      display: flex;
      align-items: center;
      padding: 10px 15px;
      background: var(--ai-bubble);
      border-radius: 18px;
      width: fit-content;
    }
    
    .typing-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--secondary);
      margin: 0 4px;
      animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-6px); }
    }
    
    /* 侧边栏切换按钮 */
    .toggle-sidebar {
      position: absolute;
      top: 20px;
      left: 20px;
      z-index: 10;
      display: none;
    }
    
    @media (max-width: 992px) {
      .toggle-sidebar {
        display: block;
      }
      
      .sidebar {
        position: absolute;
        left: -260px;
        height: calc(100vh - 40px);
        z-index: 9;
      }
      
      .sidebar.active {
        left: 0;
      }
    }
  </style>
  