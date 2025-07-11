# BikeFlow
后端的使用说明位于BikeFlow/backend/README.md。
## data
使用mysql，建库后导入database/trafiicData.sql文件即可
详情见database/README.md
## 云服务器部署
1.更改前端的anxios.js文件：baseURL: 'http://localhost:3000'为指定ip
1.更改后端的.env:CORS_ORIGIN=http://114.116.194.58:5173
2.构建前端文件
3.传输文件到后端
4.source数据库
5.启动前端、后端服务
```bash
cd /root/BikeFlow/frontend
pm2 start "serve -s dist -l 5173" --name "my-app-frontend"
cd /root/BikeFlow/backend
pm2 start app.js --name "my-app-backend"
```
