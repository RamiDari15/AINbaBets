import sys
import json
import requests
import torch
import numpy as np
import time
import torch.nn as nn

SPORTRADAR_API_URL = "https://api.sportradar.com/nba/trial/v8/en"
API_KEYS = ["HAvG13lj41gP04VAAsAxoH3W6dmtEu3QV89DmQrs"]
current_key_index = 0


def get_api_key():
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return key

def fetch_season_stats(team_id, season_year):
    url = f"{SPORTRADAR_API_URL}/seasons/{season_year}/REG/teams/{team_id}/statistics.json?api_key={get_api_key()}"
    time.sleep(5)  
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error fetching stats for team {team_id}: {e}")


def find_player_stats(team_stats, player_id):
    for player in team_stats.get("players", []):
        if player.get("id") == player_id:
            return player
    raise ValueError(f"Player ID {player_id} not found in team stats.")


def prepare_input_features(player_stats, team_stats, opponent_stats, line):
    own_stats = team_stats['own_record']['average']
    opp_stats = team_stats['opponents']['average']
    opp_own_stats = opponent_stats['own_record']['average']
    opp_opp_stats = opponent_stats['opponents']['average']
    totals = player_stats.get("total", {})
    averages = player_stats.get("average", {})
    return np.array([
        totals.get('games_played', 0),
        averages.get('rebounds', 0),
        averages.get('blocks', 0),
        averages.get('minutes', 0),
        opp_stats.get("off_rebounds", 0),
        opp_stats.get("rebounds", 0),
        own_stats.get("rebounds", 0),
        opp_stats.get("field_goals_made", 0),
        own_stats.get("blocks", 0),
        opp_own_stats.get("off_rebounds", 0),
        opp_opp_stats.get("blocks", 0), 
        line
    ])

class NBABlockModel(nn.Module):
    def __init__(self, input_dim):
        super(NBABlockModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def get_model_prediction(player_id, team_id, opponent_team_id, season_year, line):

    team_stats = fetch_season_stats(team_id, season_year)
    opponent_stats = fetch_season_stats(opponent_team_id, season_year)
    player_stats = find_player_stats(team_stats, player_id)
    input_features = prepare_input_features(player_stats, team_stats, opponent_stats, line)


    input_dim = input_features.shape[0]
    model = NBABlockModel(input_dim)
    model.load_state_dict(torch.load("best_blocks_model.pt"))
    model.eval()

    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        predicted_value = model(input_tensor).item()

    return predicted_value


def main():
    if len(sys.argv) < 6:
        print("Usage: blocks_model.py <player_id> <team_id> <opponent_team_id> <season_year> <line>")
        sys.exit(1)

    try:
        player_id = sys.argv[1]
        team_id = sys.argv[2]
        opponent_team_id = sys.argv[3]
        season_year = int(sys.argv[4])
        line = float(sys.argv[5])

        prediction = get_model_prediction(player_id, team_id, opponent_team_id, season_year, line)
        print(json.dumps({"prediction": prediction}))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()

