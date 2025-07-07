const express = require('express');
const app = express();
const PORT = 3000;

app.use("/test",require("./router/testRouter"));



app.listen(PORT, () => {
  console.log(`🚀 Server is running at http://localhost:${PORT}`);
});
