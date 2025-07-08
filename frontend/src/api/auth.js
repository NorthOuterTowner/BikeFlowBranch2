import axios from 'axios'

const request = axios.create({
  baseURL: 'http://localhost:3000',  
  timeout: 5000
})

// 用户注册
export function register(username, password) {
  return request.post('/admin/register',{
    username: username,
    password: password
  })
}

// 用户登录
export function login(username, password) {
  return request.post('/admin/login', {
    username: username,
    password: password
  })
}
