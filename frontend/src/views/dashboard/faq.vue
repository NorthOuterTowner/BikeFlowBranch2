<script setup>

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
        <h2>常见问题</h2>
        <div class="faq-item">
            <p><strong>Q：为什么我注册了没法登录呢？</strong></p>
            <p>A：请确保您输入的用户名和密码正确。注册后，用户需要自行到注册邮箱确认注册，确认后即可使用该账号密码进行登录。</p>

            <p><strong>Q：为什么我修改了密码，无法用新密码登录？</strong></p>
            <p>A：请确保您在修改密码后在注册邮箱确认修改。</p>

            <p><strong>Q：为什么我的系统显示时间与真实时间不一致？</strong></p>
            <p>A：本系统旨在支持潮汐调度模拟与回溯分析，因此当前时间是您在“系统设置”中手动选择的日期与小时，并非真实的本地时间。请前往“系统设置”确认是否设定了日期与时间。</p>

            <p><strong>Q：系统设置后是否自动保存？</strong></p>
            <p>A：不自动保存，需点击“保存”按钮以写入设置（如日期、小时等）至 localStorage。</p>
            
            <p><strong>Q:为什么我搜索的站点不是我想要的？</strong></p>
            <p>A:搜索功能支持模糊匹配，会自动选取第一个匹配的站点。建议输入站点名称的关键字，如“Central Park”或站点编号如“HB101”。</p>
            
            <p><strong>Q：图表加载时间太久？</strong></p>
            <p>A：图表加载时间可能会受到数据量的影响，尤其是在选择了较长时间范围时。请耐心等待，或尝试缩小时间范围以加快加载速度。</p>
            
            <p><strong>Q：可以导出调度方案或统计图表吗？</strong></p>
            <p>A：目前系统不支持导出调度方案或统计图表。您可以手动记录或截图相关信息。</p>
            
            <p><strong>Q：为什么站点地图上不显示任何调度箭头？</strong></p>
            <p>A：调度箭头仅在当前时间段存在调度方案时才能显示。如果无法显示，请重新设置系统时间。</p>
            
            <p><strong>Q：为什么点击退出按钮直接退出，没有确认提示？</strong></p>
            <p>A：为了避免误操作，系统在退出时已增加弹窗确认功能。若仍未弹出，请刷新页面或检查浏览器弹窗拦截设置。</p>

            
            <p><strong>Q：如何获取更多帮助或支持？</strong></p>
            <p>A：如果您在使用过程中遇到问题，请发送邮件至项目经理（23301013@bjtu.edu.cn）处反馈问题。</p>


        </div>
            
        

      </div>
    

      </main>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  position: relative;
  overflow: auto;
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

.app-main {
  flex: 1;
  overflow-y: auto;
  padding: 40px 20px;
  background-color: #f8f9fa;
  display: flex;
  justify-content: flex-start; 
  margin-bottom: 80px;
}

.content {
  max-width: 800px;
  width: 100%;
  background-color: #ffffff;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
  padding: 30px 40px;
  box-sizing: border-box;
}

.content h2 {
  font-size: 24px;
  font-weight: 600;
  color: #091275;
  margin-bottom: 30px;
  border-bottom: 2px solid #e9ecef;
  padding-bottom: 10px;
}

.faq-item {
  display: flex;
  flex-direction: column;
  gap: 25px;
}

.faq-item p {
  margin: 0;
  line-height: 1.6;
  font-size: 16px;
  color: #333;
}

.faq-item strong {
  display: block;
  font-size: 17px;
  color: #0a1580;
  margin-bottom: 6px;
}


</style>