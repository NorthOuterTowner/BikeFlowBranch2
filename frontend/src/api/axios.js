import axios from 'axios'

const request = axios.create({
  baseURL: 'http://localhost:3000',  // 如果 vite.config.js 配了 proxy，就保持 '/'；否则用后端地址
  timeout: 5000
})

request.interceptors.request.use(config => {
  const token = sessionStorage.getItem('token')
  const account = sessionStorage.getItem('account')
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

export function getDispatch(queryTime) {
  return request.get('/dispatch', {
    params: { query_time: queryTime }
  })     
}

export function startDispatch(data) {
  return request.post('/dispatch/change', data)
}

export function cancelDispatch(data) {
  return request.post('/dispatch/cancelChange', data)
}

export async function getStationAssign(params = {}) {
  try {
    const res = await request.get('/search/stationAssign', { params });
    console.log('调出站点接口返回:', res.data);
    const result = res.data.station_result;
    if (!result || !Array.isArray(result)) {
      console.error('获取调出站点失败，返回数据格式错误', res.data);
      return [];
    }
    return result;
  } catch (error) {
    console.error('获取调出站点接口请求失败', error);
    return [];
  }
}

export default request
