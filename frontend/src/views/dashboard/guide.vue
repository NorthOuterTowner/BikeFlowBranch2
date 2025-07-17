<script setup>
const welcoming = '管理员，欢迎您！'
const logout = () => {
  alert('已退出登录')
}
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <h1 class="title">共享单车调度系统使用教程</h1>
      </div>
      <div class="user-info">
        <div class="user-top">
          <span class="welcoming">{{ welcoming }}</span>
          <button class="logout-button" @click="logout">退出</button>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="app-main">
      <div class="content">
        <div class="guidPart" v-for="(part, index) in guideParts" :key="index">
          <h2>{{ index + 1 }}. {{ part.title }}</h2>
          <div v-for="(item, idx) in part.contents" :key="idx" class="section">
            <img v-if="item.img" :src="item.img" alt="image" class="guid-image" />
            <p v-if="item.text">{{ item.text }}</p>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
export default {
  data() {
    return {
      guideParts: [
        {
          title: '登录注册',
          contents: [
            { img: '/icons/register.png', text: '在使用本系统前，用户需要先注册账号。' },
            { img: '/icons/confirm.png', text: '注册后，用户需要自行到注册邮箱确认注册，确认后即可使用该账号密码进行登录。' },
            { img: '/icons/login.png' }
          ]
        },
        {
          title: '系统设置',
          contents: [
            { img: '/icons/systemSetting.png', text: '用户可以选择日期和小时，系统会加载对应时间段数据。点击“保存”按钮写入设置至 localStorage。' }
          ]
        },
        {
          title: '个人信息',
          contents: [
            { img: '/icons/profile.png', text: '用户可以查看与修改用户名和密码（修改密码需邮箱确认）。' }
          ]
        },
        {
          title: '预测显示',
          contents: [
            { img: '/icons/mapView.png', text: '查看各站点潮汐预测情况，并可通过ID或名称搜索。' },
            { img: '/icons/sInf.png', text: '查看热力图展示的潮汐预测情况。' },
            { img: '/icons/heatMap.png' }
          ]
        },
        {
          title: '调度方案',
          contents: [
            { img: '/icons/schedule.png', text: '查看当前时间段的调度方案及站点情况。' },
            { img: '/icons/sInf2.png', text: '调度详情页面支持采纳、拒绝、修改调度数量及导航查看。' },
            { img: '/icons/dispatch.png' }
          ]
        },
        {
          title: '与DeepSeek对话',
          contents: [
            { img: '/icons/deepseek.png', text: '用户可与DeepSeek对话，获取调度建议并重新生成调度方案。' }
          ]
        },
        {
          title: '数据统计',
          contents: [
            { img: '/icons/statistics.png', text: '查看站点潮汐预测数据，并了解调度情况。' }
          ]
        }
      ]
    }
  }
}
</script>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh; /* 保证整体视口高度 */
  overflow: hidden; /* 防止整体页面有两个滚动条 */
}

.app-main {
  flex: 1; /* 自动撑满剩余高度 */
  overflow-y: auto; /* 关键：启用垂直滚动 */
  padding: 20px;
  box-sizing: border-box;
  margin-bottom: 80px;
}

.app-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 10px 20px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ccc;
  flex-shrink: 0;
  width: 100%;
  z-index: 50;
  box-sizing: border-box;
}

.header-left {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex: 1;
  min-width: 0;
}

.title {
  font-size: 20px;
  font-weight: bold;
  margin: 0;
}
.user-info {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  margin-left: 20px;
  gap: 15px;
  flex-shrink: 0;
}

.user-top {
  display: flex;
  align-items: center;
  gap: 20px;
}

.welcoming {
  font-size: 14px;
  white-space: nowrap;
  color: #495057;
}

.logout-button{
  padding: 6px 12px;
  background-color: #091275;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  white-space: nowrap;
  transition: background-color 0.2s;
}

.logout-button:hover {
  background-color: #0a1580;
}


.content {
  max-width: 1000px;
  margin: 0 auto;
}

.guidPart {
  background: white;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 30px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.guidPart h2 {
  font-size: 20px;
  color: #091275;
  margin-bottom: 16px;
  border-bottom: 1px solid #eee;
  padding-bottom: 8px;
}

.guid-image {
  width: 100%;
  max-height: 400px;
  object-fit: contain;
  margin: 16px 0;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.section p {
  font-size: 15px;
  color: #444;
  margin-bottom: 16px;
  line-height: 1.6;
}
</style>
