const autocannon = require('autocannon');

autocannon({
  url: 'http://localhost:3000/stations/bikeNum/timeAll?date=2025-06-13&hour=9',
  method: 'GET',
  connections: 50,
  duration: 5,
  headers: {
    'Content-Type': 'application/json',
    'account': 'admin',
    'token': 'adef1074-ed28-42a7-b47f-bb40801352f9'
  }
}, (err, result) => {
  if (err) {
    console.error('压测失败:', err);
  } else {
    console.log('压测完成');
    console.log(result);
  }
});
