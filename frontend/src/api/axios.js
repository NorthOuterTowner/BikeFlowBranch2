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

// 用户注册
export function register(username, password, email) {
  return request.post('/admin/register',{
    account: username,
    password: password,
    email:email
  }, {
    headers: { 'Content-Type': 'application/json' }  // 👈 确保是 json
  })
}

// 用户登录
export function login(username, password) {
  return request.post('/admin/login', {
    account: username,
    password: password
  })
}

export default request
