<template>
    <aside class="sidebar">
      <nav>
        <h1>预测显示</h1>
        <ul>
          <li><router-link to="/dashboard/mapView">地图</router-link></li>
          <li><router-link to="/dashboard/predict">预测结果</router-link></li>
        </ul>
  
        <h1>调度显示</h1>
        <ul>
          <li><router-link to="/dashboard/schedule">调度页面</router-link></li>
          <li><router-link to="/dashboard/dispatch">调度详情</router-link></li>
        </ul>
  
        <h1>设置</h1>
        <ul>
          <li><router-link to="/dashboard/settings">系统设置</router-link></li>
          <li><router-link to="/dashboard/profile">个人信息</router-link></li>
        </ul>
      </nav>
    </aside>
  </template>
  
<style scoped>
  .sidebar {
    width: 200px;
    background-color: #f5f5f5;
    padding: 1rem;
  }
  .sidebar h1 {
    font-size: 1rem;
    margin: 1rem 0 0.5rem;
  }
  .sidebar ul {
    list-style: none;
    padding-left: 0;
  }
  .sidebar li {
    margin-bottom: 0.5rem;
  }
</style>
  