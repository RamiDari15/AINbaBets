import { spawn } from "child_process";


const calculateStat = async (scriptName, playerId, teamId, opponentTeamId, seasonYear, line) => {
  return new Promise((resolve, reject) => {
    const process = spawn("python3", [`./${scriptName}`, playerId, teamId, opponentTeamId, seasonYear, line]);

    let output = "";
    let error = "";

    process.stdout.on("data", (data) => {
      output += data.toString();
    });

    process.stderr.on("data", (data) => {
      error += data.toString();
    });

    process.on("close", (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(output); 
          resolve(result.prediction); 
        } catch (err) {
          console.error("Error parsing Python output:", err);
          reject(`Error parsing Python output: ${err}`);
        }
      } else {
        console.error("Python process error:", error);
        reject(`Python process exited with code ${code}: ${error}`);
      }
    });
  });
};



const handlePrediction = async (leg) => {
  const { player, team_id, opponent_team_id, stat, value, overUnder } = leg;

  if (!value || isNaN(value)) {
    throw new Error(`Invalid value (line) for leg: ${JSON.stringify(leg)}`);
  }



  switch (stat) {
    case "points":
      const pointsPrediction = await calculateStat("points_model_api.py", player, team_id, opponent_team_id, 2024, parseFloat(value));
      return overUnder === "under" ? 1 - pointsPrediction : pointsPrediction;

    case "assists":
      const assistsPrediction = await calculateStat("assists_model_api.py", player, team_id, opponent_team_id, 2024, parseFloat(value));
      return overUnder === "under" ? 1 - assistsPrediction : assistsPrediction;

    case "rebounds":
      const reboundsPrediction = await calculateStat("rebounds_model_api.py", player, team_id, opponent_team_id, 2024, parseFloat(value));
      return overUnder === "under" ? 1 - reboundsPrediction : reboundsPrediction;

    case "blocks":
      const blocksPrediction = await calculateStat("blocks_model_api.py", player, team_id, opponent_team_id, 2024, parseFloat(value));
      return overUnder === "under" ? 1 - blocksPrediction : blocksPrediction;

    case "steals":
      const stealsPrediction = await calculateStat("steals_model_api.py", player, team_id, opponent_team_id, 2024, parseFloat(value));
      return overUnder === "under" ? 1 - stealsPrediction : stealsPrediction;

    default:
      throw new Error(`Stat type "${stat}" is not supported.`);
  }
};


const calculateParlay = async (legs) => {
  if (!Array.isArray(legs) || legs.length === 0) {
    throw new Error("No legs provided for the parlay.");
  }

  try {
    const legPredictions = await Promise.all(
      legs.map(async (leg) => {
        try {
          const prediction = await handlePrediction(leg);
          return prediction;
        } catch (error) {
          console.error(`Error processing leg ${JSON.stringify(leg)}: ${error}`);
          return null;
        }
      })
    );

    const validPredictions = legPredictions.filter((pred) => pred !== null);

    if (validPredictions.length === 0) {
      throw new Error("No valid predictions could be calculated for the parlay.");
    }

    let parlayPercentage = validPredictions.reduce((acc, pred) => acc * pred, 1);


    console.log(`Parlay Combined Percentage: ${parlayPercentage}`);
    return parlayPercentage;
  } catch (error) {
    console.error(`Error calculating parlay: ${error}`);
    throw error;
  }
};

export { calculateParlay };
