import requests
import torch
import torch.nn as nn
import numpy as np
import time

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
    try:
        response = requests.get(url)
        response.raise_for_status()
        time.sleep(3)  # Pause to avoid rate limiting
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for team {team_id}: {e}")
        return None

def find_player_stats(team_stats, player_id):
    for player in team_stats.get("players", []):
        if player.get("id") == player_id:
            return player
    raise ValueError(f"Player ID {player_id} not found in team stats.")

\
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
        own_stats.get('off_rebounds', 0),
        opp_stats.get('off_rebounds', 0),
        own_stats.get("def_rebounds", 0),
        opp_stats.get("def_rebounds", 0),
        own_stats.get("rebounds", 0),
        opp_stats.get("rebounds", 0),
        opp_stats.get("field_goals_made", 0),
        own_stats.get("blocks", 0),
        opp_own_stats.get('off_rebounds', 0),
        opp_opp_stats.get('off_rebounds', 0),
        opp_own_stats.get("def_rebounds", 0),
        opp_opp_stats.get("def_rebounds", 0),
        opp_own_stats.get("rebounds", 0),
        opp_opp_stats.get("rebounds", 0),
        opp_opp_stats.get("field_goals_made", 0),
        opp_own_stats.get("blocks", 0),
        line  
    ])


class NBAReboundModel(nn.Module):
    def __init__(self, input_dim):
        super(NBAReboundModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def get_model_prediction(player_id, team_id, opponent_team_id, season_year, line):
    team_stats = fetch_season_stats(team_id, season_year)
    opponent_stats = fetch_season_stats(opponent_team_id, season_year)

    if not team_stats or not opponent_stats:
        raise ValueError("Failed to fetch team or opponent stats.")


    player_stats = find_player_stats(team_stats, player_id)
    input_features = prepare_input_features(player_stats, team_stats, opponent_stats, line)
    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

    input_dim = input_features.shape[0]
    model = NBAReboundModel(input_dim)
    model.load_state_dict(torch.load("best_rebound_model.pt"))
    model.eval()

    with torch.no_grad():
        predicted_value = model(input_tensor).item()
        return {"prediction": predicted_value}

def main():
    player_id = "37fbc3a5-0d10-4e22-803b-baa2ea0cdb12"
    team_id = "583eca2f-fb46-11e1-82cb-f4ce4684ea4c"
    opponent_team_id = "583ecae2-fb46-11e1-82cb-f4ce4684ea4c"
    season_year = 2024
    line = 2

    try:
        result = get_model_prediction(player_id, team_id, opponent_team_id, season_year, line)
        print(f"Predicted Value: {result['prediction']}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
