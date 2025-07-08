<template>
    <div id="cesiumContainer" style="width: 100%; height: 100%;"></div>
</template>
 

<script setup>
import { onMounted } from 'vue'

onMounted(() => {
  // 设置cesium token
  Cesium.Ion.defaultAccessToken = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI0N2Y4MzU3NS00NDIxLTQyNjgtYjY2OC0yYzAzOWYxYmUwOWUiLCJpZCI6MzE5Mzk4LCJpYXQiOjE3NTE5NDEwMDZ9.HiRi9vYVH_oyH0UyFAqIiJRNIdYa0fp_nssoNSmyNTY"
  var viewer = new Cesium.Viewer("cesiumContainer", {
    // 去掉对话框警告
    infoBox: false,
    // 是否显示查询按钮
    geocoder: false,
    // 不显示home按钮
    homeButton: false,
    // 控制查看器的显示模式（3D、2.5D、2D是否显示）
    sceneModePicker: false,
    // 是否显示图层选择
    baseLayerPicker: false,
    // 是否显示帮助按钮
    navigationHelpButton: false,
    // 是否显示播放动画、计时
    animation: false,
    // 是否显示时间轴
    timeline: false,
    // 是否显示全屏按钮
    fullscreenButton: false,
  });
  // 隐藏图标
  viewer.cesiumWidget.creditContainer.style.display = "none";

  // 设置cesium默认视角
  Cesium.Camera.DEFAULT_VIEW_RECTANGLE = Cesium.Rectangle.fromDegrees(
    // 西经
    115.8651,
    // 南维度
    28.7019,
    // 东经
    115.8709,
    // 北纬
    28.7053
  )
})
</script>