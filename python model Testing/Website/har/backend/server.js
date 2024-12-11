import express from "express";
import router from "./players.js"; 
import cors from "cors";
import parlayRouter from "./parlay.js";

const app = express();
const PORT = 5002;

app.use(cors());
app.use(express.json());


app.use("/players", router);
app.use("/parlay", parlayRouter);



app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
