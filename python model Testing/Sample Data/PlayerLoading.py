import requests
import time
from datetime import datetime


SPORTRADAR_API_URL = "https://api.sportradar.com/nba/trial/v8/en"
API_KEYS = ["dGsHp38O2tgP7rrxlU5gSuE5RiAGL3vMA2TcSs9A"]
current_key_index = 0

# Function rotate API keys
def get_api_key():
    global current_key_index
    key = API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    return key


def fetch_daily_schedule(date):
    url = f"{SPORTRADAR_API_URL}/games/{date}/schedule.json?api_key={get_api_key()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching daily schedule: {e}")
        return None

def fetch_team_roster(team_id, season_year):
    url = f"{SPORTRADAR_API_URL}/seasons/{season_year}/REG/teams/{team_id}/statistics.json?api_key={get_api_key()}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("players", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching roster for team {team_id}: {e}")
        return []


def get_players_for_date(date, season_year):
    schedule = fetch_daily_schedule(date)
    if not schedule or "games" not in schedule:
        print("Failed to fetch the daily schedule.")
        return []

    games = schedule["games"]
    all_players = []

    for game in games:
        home_team_id = game.get("home", {}).get("id")
        away_team_id = game.get("away", {}).get("id")

        # Fetch players for home team
        if home_team_id:
            home_team_players = fetch_team_roster(home_team_id, season_year)
            for player in home_team_players:
                all_players.append({
                    "player_id": player["id"],
                    "name": f"{player['first_name']} {player['last_name']}"
                })


        if away_team_id:
            away_team_players = fetch_team_roster(away_team_id, season_year)
            for player in away_team_players:
                all_players.append({
                    "player_id": player["id"],
                    "name": f"{player['first_name']} {player['last_name']}"
                })

        time.sleep(10)

    # Filter duplicates 
    unique_players = {player["player_id"]: player for player in all_players}.values()

    return list(unique_players)


if __name__ == "__main__":
    date = datetime.now().strftime("%Y/%m/%d")  # Get current date in YYYY-MM-DD format
    season_year = datetime.now().year  # Current NBA season year
    players_data = get_players_for_date(date, season_year)
    

    print("Players for the day:")
    for player in players_data:
        print(f"Player ID: {player['player_id']}, Name: {player['name']}")
