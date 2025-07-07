const express = require('express');
const app = express();
const PORT = 3000;
const {db} = require("./db/dbUtils")

/*Cross-Origin Requests */
app.use(function(req,res,next){
    res.header("Access-Control-Allow-Origin","*");
    res.header("Access-Control-Allow-Headers","*");
    res.header("Access-Control-Allow-Methods","DELETE,PUT,POST,GET,OPTIONS");
    if(req.method == "OPTIONS") res.sendStatus(200);
    else next();
});

app.use("/test",require("./router/testRouter"));
app.use("/login",require("./router/adminRouter"));


app.listen(PORT, () => {
  console.log(`🚀 Server is running at http://localhost:${PORT}`);
});
