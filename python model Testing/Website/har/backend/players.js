import express from "express";
import axios from "axios";

const router = express.Router();

const SPORTRADAR_API_URL = "https://api.sportradar.com/nba/trial/v8/en";
const API_KEY = "HAvG13lj41gP04VAAsAxoH3W6dmtEu3QV89DmQrs"; 

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

router.get("/players", async (req, res) => {
  const date = new Date();
  const formattedDate = `${date.getFullYear()}/${date.getMonth() + 1}/${date.getDate()}`;
  const seasonYear = 2024; 

  try {

    const scheduleResponse = await axios.get(
      `${SPORTRADAR_API_URL}/games/${formattedDate}/schedule.json?api_key=${API_KEY}`
    );
    const games = scheduleResponse.data.games || [];

    const players = [];

    for (const game of games) {
      const { home, away } = game;

      if (home?.id) {
        const homePlayersResponse = await axios.get(
          `${SPORTRADAR_API_URL}/seasons/${seasonYear}/REG/teams/${home.id}/statistics.json?api_key=${API_KEY}`
        );
        players.push(
          ...homePlayersResponse.data.players.map((player) => ({
            player_id: player.id,
            name: `${player.first_name} ${player.last_name}`,
            team_id: home.id,
            opponent_team_id: away?.id || null, // Opponent team ID
          }))
        );
        await delay(2500); 
      }


      if (away?.id) {
        const awayPlayersResponse = await axios.get(
          `${SPORTRADAR_API_URL}/seasons/${seasonYear}/REG/teams/${away.id}/statistics.json?api_key=${API_KEY}`
        );
        players.push(
          ...awayPlayersResponse.data.players.map((player) => ({
            player_id: player.id,
            name: `${player.first_name} ${player.last_name}`,
            team_id: away.id,
            opponent_team_id: home?.id || null, 
          }))
        );
        await delay(2500); 
      }
    }


    const uniquePlayers = Array.from(
      new Map(players.map((p) => [p.player_id, p])).values()
    );

    res.json(uniquePlayers);
  } catch (error) {
    console.error("Error fetching players:", error);
    res.status(500).send("Server error.");
  }
});

export default router;
