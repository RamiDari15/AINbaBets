import requests
import torch
import torch.nn as nn
import time

# Define the Sportradar API endpoint and keys
SPORTRADAR_API_URL = "https://api.sportradar.com/nba/trial/v8/en"
API_KEYS = [
    "dGsHp38O2tgP7rrxlU5gSuE5RiAGL3vMA2TcSs9A"
]
current_key_index = 0


# Function to rotate API keys
def get_api_key():
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return key


# Function to fetch player profile
def fetch_player_profile(player_id):
    time.sleep(5)  # Throttle API calls
    url = f"{SPORTRADAR_API_URL}/players/{player_id}/profile.json?api_key={get_api_key()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching player profile for {player_id}: {e}")
        return None


# Function to fetch team profile
def fetch_team_profile(team_id):
    time.sleep(5)  # Throttle API calls
    url = f"{SPORTRADAR_API_URL}/teams/{team_id}/profile.json?api_key={get_api_key()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team profile for {team_id}: {e}")
        return None


# Function to fetch daily schedule
def fetch_daily_schedule(date):
    time.sleep(5)  # Throttle API calls
    url = f"{SPORTRADAR_API_URL}/games/{date}/schedule.json?api_key={get_api_key()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching daily schedule for {date}: {e}")
        return None


# PyTorch model class
class NBAPointModel(nn.Module):
    def __init__(self, input_size):
        super(NBAPointModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


# Main function to collect model inputs and make predictions
def points_method(player_id, game_date, weights_file_path):
    try:
        # Fetch player profile to get the player's team ID
        player_profile = fetch_player_profile(player_id)
        if not player_profile or "team" not in player_profile:
            raise ValueError("Failed to fetch player profile or team ID")
        player_team_id = player_profile["team"]["id"]

        # Fetch daily schedule to find the opposing team
        daily_schedule = fetch_daily_schedule(game_date)
        if not daily_schedule or "games" not in daily_schedule:
            raise ValueError("Failed to fetch daily schedule or games")

        game = next(
            (g for g in daily_schedule["games"] if g["home"]["id"] == player_team_id or g["away"]["id"] == player_team_id),
            None
        )
        if not game:
            raise ValueError("Player's team game not found in the daily schedule")

        opponent_team_id = game["away"]["id"] if game["home"]["id"] == player_team_id else game["home"]["id"]

        # Fetch team profiles for player's team and opponent team
        player_team_profile = fetch_team_profile(player_team_id)
        opponent_team_profile = fetch_team_profile(opponent_team_id)

        if not player_team_profile or not opponent_team_profile:
            raise ValueError("Failed to fetch team profiles")

        # Extract model inputs
        model_inputs = [
            player_profile["statistics"]["points"],
            player_profile["statistics"]["games_played"],
            player_profile["statistics"]["points_per_game"],
            player_profile["statistics"]["field_goals_made"],
            player_profile["statistics"]["field_goals_attempted"],
            player_profile["statistics"]["field_goal_percentage"],
            player_profile["statistics"]["three_point_made"],
            player_profile["statistics"]["three_point_attempted"],
            player_profile["statistics"]["three_point_percentage"],
            player_profile["statistics"]["free_throws_made"],
            player_profile["statistics"]["free_throws_attempted"],
            player_profile["statistics"]["free_throw_percentage"],
            player_profile["statistics"]["true_shooting_percentage"],
            player_profile["statistics"]["usage_rate"],
            player_profile["statistics"]["minutes_per_game"],
            player_team_profile["statistics"]["fast_break_points"],
            player_team_profile["statistics"]["points_off_turnovers"],
            player_team_profile["statistics"]["points"],
            player_team_profile["statistics"]["rebounds"],
            player_team_profile["statistics"]["turnovers"],
            player_team_profile["statistics"]["opponent_turnovers"],
            opponent_team_profile["statistics"]["fast_break_points"],
            opponent_team_profile["statistics"]["points_off_turnovers"],
            opponent_team_profile["statistics"]["points"],
            opponent_team_profile["statistics"]["turnovers"],
            opponent_team_profile["statistics"]["opponent_turnovers"],
        ]

        # Load PyTorch model
        model = NBAPointModel(len(model_inputs))
        model.load_state_dict(torch.load(weights_file_path))
        model.eval()

        # Prepare input tensor
        input_tensor = torch.tensor(model_inputs, dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor.unsqueeze(0))
            predicted_probability = prediction.item()

        print("Prediction:", predicted_probability)
        return predicted_probability
    except Exception as e:
        print("Error making model prediction:", e)
        return None


# Example usage
if __name__ == "__main__":
    player_id = "823b2161-0c34-494c-9d7c-b438152f4f4d"
    game_date = "2024/12/02"
    weights_file_path = "best_points_model.pt"
    points_method(player_id, game_date, weights_file_path)
