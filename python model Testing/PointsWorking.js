import { getModelPrediction } from "./PointsMethod";

const predictPlayerPerformance = async () => {
  const playerId = "823b2161-0c34-494c-9d7c-b438152f4f4d";
  const gameDate = "2023-12-02"; 
  const weightsFilePath = "../best_points_model.pt"; 

  const prediction = await getModelPrediction(playerId, gameDate, weightsFilePath);
  console.log("Prediction:", prediction);
};

predictPlayerPerformance();
