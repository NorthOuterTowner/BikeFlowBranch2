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
    headers: { 'Content-Type': 'application/json' }  //  确保是 json
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

export async function postSuggestion(target_time,message) {
  const account = sessionStorage.getItem('account') 
  const token = sessionStorage.getItem('token')
  console.log('即将使用的 token：', token)
  try {
    const res = await fetch('http://localhost:3000/suggestions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        account: account,   // 'admin' 也可以直接写死
        token: token
      },
      body: JSON.stringify({ target_time,message })
    })

    if (!res.ok) {
      console.error('请求失败：HTTP 状态码', res.status)
      return null
    }

    const data = await res.json()
    console.log('后端返回数据：', data)
    if (!data || typeof data !== 'object' || !data.suggestion) {
      console.error('接口返回格式不符合预期：', data)
      return null
    }
    return data.suggestion
  } catch (error) {
    console.error('请求出错', error)
    return null
  }
}

export async function postDispatchPlan(target_time, user_guidance) {
  const account = sessionStorage.getItem('account')
  const token = sessionStorage.getItem('token')
  console.log('即将使用的 token：', token)

  try {
    const res = await fetch('http://localhost:3000/suggestions/dispatch', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        account: account,
        token: token
      },
      body: JSON.stringify({ target_time, user_guidance })
    })

    if (!res.ok) {
      console.error('请求失败：HTTP 状态码', res.status)
      return `请求失败：HTTP 状态码 ${res.status}`
    }

    const data = await res.json()
    console.log('后端返回数据：', data)

    if (!data || typeof data !== 'object' || !Array.isArray(data.optimized_plan)) {
      console.error('接口返回格式不符合预期：', data)
      return '后端返回格式不符合预期'
    }

    // 格式化可读文本
    return data // 返回原始 json：{ schedule_time, optimized_plan }
  } catch (error) {
    console.error('请求出错', error)
    return '请求出错，请稍后再试'
  }
}



export default request
