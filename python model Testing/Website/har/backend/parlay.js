import express from "express";
import { calculateParlay } from "./CalcOutput.js"; 

const router = express.Router();


router.post("/calculate", async (req, res) => {
    try {
      const { legs } = req.body;
      if (!legs || !Array.isArray(legs)) {
        return res.status(400).json({ error: "Invalid or missing 'legs' in request body" });
      }
  

      const parlayPercentage = await calculateParlay(legs);
      res.json({ parlayPercentage });
    } catch (error) {
      console.error("Error calculating parlay:", error);
      res.status(500).json({ error: "Internal Server Error" });
    }
  });


export default router;
