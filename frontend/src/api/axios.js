import axios from 'axios'

const request = axios.create({
  baseURL: 'http://localhost:3000',  // 如果 vite.config.js 配了 proxy，就保持 '/'；否则用后端地址
  timeout: 5000
})

request.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  const account = localStorage.getItem('account')
  if (token && account) {
    config.headers['token'] = token
    config.headers['account'] = account
  }
  return config
}, error => Promise.reject(error))

export default request
