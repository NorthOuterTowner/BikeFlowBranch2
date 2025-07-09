const express = require('express');
const app = express();
const PORT = 3000;
const {db,genid} = require("./db/dbUtils")

/*Cross-Origin Requests */
app.use(function(req,res,next){
    res.header("Access-Control-Allow-Origin","*");
    res.header("Access-Control-Allow-Headers","*");
    res.header("Access-Control-Allow-Methods","DELETE,PUT,POST,GET,OPTIONS");
    if(req.method == "OPTIONS") res.sendStatus(200);
    else next();
});

app.use(express.json());

app.use("/test",require("./router/testRouter"));
app.use("/admin",require("./router/adminRouter"));
app.use("/reset",require("./router/resetRouter"));
app.use("/stations",require("./router/stationsRouter"));
app.use("/predict",require("./router/predictRouter"));

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
