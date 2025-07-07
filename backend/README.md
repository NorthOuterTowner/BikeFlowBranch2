# Backend使用说明

## 简要说明
backend部分 是一个基于 Node.js 的后端应用程序，旨在根据需求为前端提供相关 API。该项目使用 Express 框架构建，并集成了 MySQL 数据库。

## 安装依赖
1. 安装 Node.js 和 npm
   - 请参考 [Node.js 官网](https://nodejs.org/) 下载并安装最新版本的 Node.js。
2. 安装依赖（这一步最好在系统cmd中以管理员身份运行）
    切换到项目目录下的 `BikeFlow/backend` 文件夹，并使用 npm 安装所需的依赖包。
   ```bash
    cd BikeFlow/backend
    npm install
    ```
3. 启动app.js（在VScode的集成终端运行即可）
    ```bash
     node app.js
     ```
    若终端显示🚀 Server is running at http://localhost:${PORT}，则说明后端启动成功。
4. 结束进程
   - 在终端中按 `Ctrl + C` 结束当前进程。
