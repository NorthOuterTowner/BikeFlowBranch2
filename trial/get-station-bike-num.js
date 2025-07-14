const autocannon = require('autocannon');

autocannon({
  url: 'http://localhost:3000/stations/bikeNum/timeAll?date=2025-06-13&hour=9',
  method: 'GET',
  connections: 500,
  duration: 60,
  headers: {
    'Content-Type': 'application/json',
    'account': 'admin',
    'token': 'c012de5d-6948-4aef-af17-a2e587ebe826'
  }
}, (err, result) => {
  if (err) {
    console.error('压测失败:', err);
  } else {
    console.log('压测完成');
    console.log(result);
  }
});
