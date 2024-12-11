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
        time.sleep(5)  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for team {team_id}: {e}")
        return None

def find_player_stats(team_stats, player_id):
    for player in team_stats.get("players", []):
        if player.get("id") == player_id:
            return player
    raise ValueError(f"Player ID {player_id} not found in team stats.")

def prepare_input_features(player_stats, team_stats, opponent_stats):
    own_stats = team_stats['own_record']['average']
    opp_stats = team_stats['opponents']['average']
    opp_own_stats = opponent_stats['own_record']['average']
    opp_opp_stats = opponent_stats['opponents']['average']
    totals = player_stats.get("total", {})
    averages = player_stats.get("average", {})

    return np.array([
        totals.get('games_played', 0),
        averages.get('assists', 0),
        averages.get('turnovers', 0),
        totals.get('usage_pct', 0),
        averages.get('minutes', 0),
        own_stats.get("fast_break_pts", 0),
        own_stats.get("assists", 0),
        opp_own_stats.get("assists", 0),
        0
    ])

# Model Definition
class NBAAssistModel(nn.Module):
    def __init__(self, input_dim):
        super(NBAAssistModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  
        self.fc4 = nn.Linear(32, 16)  
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))  
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc5(x))
        return x


def get_model_prediction(player_id, team_id, opponent_team_id, season_year):
    team_stats = fetch_season_stats(team_id, season_year)
    opponent_stats = fetch_season_stats(opponent_team_id, season_year)

    if not team_stats or not opponent_stats:
        raise ValueError("Failed to fetch team or opponent stats.")

    player_stats = find_player_stats(team_stats, player_id)


    input_features = prepare_input_features(player_stats, team_stats, opponent_stats)

    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)

    input_dim = input_features.shape[0]
    model = NBAAssistModel(input_dim)
    model.load_state_dict(torch.load("best_assist_model.pt", weights_only=True))
    model.eval()

    with torch.no_grad():
        predicted_value = model(input_tensor).item()
        binary_prediction = 1 if predicted_value >= 0.5 else 0
        print(f"Predicted Probability: {predicted_value}")
        print(f"Binary Prediction: {binary_prediction}")
        return binary_prediction


if __name__ == "__main__":
    player_id = "460b7264-b98f-483e-b841-59a18c2e4d67"
    team_id = "583eca2f-fb46-11e1-82cb-f4ce4684ea4c"
    opponent_team_id = "583ecae2-fb46-11e1-82cb-f4ce4684ea4c"
    season_year = 2023

    prediction = get_model_prediction(player_id, team_id, opponent_team_id, season_year)
    print(f"Predicted Value: {prediction}")
