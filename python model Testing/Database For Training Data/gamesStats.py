import sqlite3
import requests
import time


RAPIDAPI_KEY = "1389ba43e2msh64531277a8c72a6p13fafajsn708d2f89b52e"  # Replace with your RapidAPI key
BASE_URL = "https://api-nba-v1.p.rapidapi.com"
HEADERS = {
    "X-RapidAPI-Key": RAPIDAPI_KEY,
    "X-RapidAPI-Host": "api-nba-v1.p.rapidapi.com"
}
DELAY = .21 


conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()



def create_games_table():

    cursor.execute('''
        CREATE TABLE Game (
            game_id TEXT PRIMARY KEY,
            season INTEGER,
            date TEXT,
            home_team_id INTEGER,
            home_team_name TEXT,
            visitors_team_id INTEGER,
            visitors_team_name TEXT,
            home_team_score INTEGER,
            visitors_team_score INTEGER
        );
    ''')
    conn.commit()

def create_player_game_stats_table(year):
    table_name = f"PlayerGameStats_{year}"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            game_id TEXT,
            player_id INTEGER,
            firstname TEXT,
            lastname TEXT,
            team_id INTEGER,
            team_name TEXT,
            position TEXT,
            minutes TEXT,
            points INTEGER,
            fgm INTEGER,
            fga INTEGER,
            fgp REAL,
            ftm INTEGER,
            fta INTEGER,
            ftp REAL,
            tpm INTEGER,
            tpa INTEGER,
            tpp REAL,
            offReb INTEGER,
            defReb INTEGER,
            totReb INTEGER,
            assists INTEGER,
            pFouls INTEGER,
            steals INTEGER,
            turnovers INTEGER,
            blocks INTEGER,
            plusMinus REAL,
            PRIMARY KEY (game_id, player_id),
            FOREIGN KEY (game_id) REFERENCES Games (game_id)
        );
    ''')
    conn.commit()

def fetch_games(season):
    url = f"{BASE_URL}/games"
    params = {"season": season}
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    data = response.json()
    games = data.get("response", [])
    return games

def insert_games(games, season):
    for game in games:
        game_data = {
            "game_id": game["id"],
            "season": season,
            "date": game["date"]["start"],
            "home_team_id": game["teams"]["home"]["id"],
            "home_team_name": game["teams"]["home"]["name"],
            "visitors_team_id": game["teams"]["visitors"]["id"],
            "visitors_team_name": game["teams"]["visitors"]["name"],
            "home_team_score": game["scores"].get("home", {}).get("points", 0),
            "visitors_team_score": game["scores"].get("visitors", {}).get("points", 0)
        }
        
        cursor.execute('''
            INSERT OR REPLACE INTO Game (
                game_id, season, date, home_team_id, home_team_name,
                visitors_team_id, visitors_team_name, home_team_score, visitors_team_score
            ) VALUES (
                :game_id, :season, :date, :home_team_id, :home_team_name,
                :visitors_team_id, :visitors_team_name, :home_team_score, :visitors_team_score
            );
        ''', game_data)
    conn.commit()


def fetch_player_stats(game_id):
    url = f"{BASE_URL}/players/statistics"
    params = {"game": game_id}

    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()
    players = data.get("response", [])
    return players


def insert_player_stats(players, year, game_id):
    table_name = f"PlayerGameStats_{year}"
    for player in players:
        stats = player
        player_data = {
            "game_id": game_id,
            "player_id": stats["player"]["id"],
            "firstname": stats["player"]["firstname"],
            "lastname": stats["player"]["lastname"],
            "team_id": stats["team"]["id"],
            "team_name": stats["team"]["name"],
            "position": stats.get("pos", ""),
            "minutes": stats.get("min", ""),
            "points": stats.get("points", 0),
            "fgm": stats.get("fgm", 0),
            "fga": stats.get("fga", 0),
            "fgp": stats.get("fgp", 0.0),
            "ftm": stats.get("ftm", 0),
            "fta": stats.get("fta", 0),
            "ftp": stats.get("ftp", 0.0),
            "tpm": stats.get("tpm", 0),
            "tpa": stats.get("tpa", 0),
            "tpp": stats.get("tpp", 0.0),
            "offReb": stats.get("offReb", 0),
            "defReb": stats.get("defReb", 0),
            "totReb": stats.get("totReb", 0),
            "assists": stats.get("assists", 0),
            "pFouls": stats.get("pFouls", 0),
            "steals": stats.get("steals", 0),
            "turnovers": stats.get("turnovers", 0),
            "blocks": stats.get("blocks", 0),
            "plusMinus": stats.get("plusMinus", 0.0)
        }
        cursor.execute(f'''
            INSERT OR REPLACE INTO {table_name} (
                game_id, player_id, firstname, lastname, team_id, team_name,
                position, minutes, points, fgm, fga, fgp, ftm, fta, ftp, tpm,
                tpa, tpp, offReb, defReb, totReb, assists, pFouls, steals,
                turnovers, blocks, plusMinus
            ) VALUES (
                :game_id, :player_id, :firstname, :lastname, :team_id, :team_name,
                :position, :minutes, :points, :fgm, :fga, :fgp, :ftm, :fta, :ftp,
                :tpm, :tpa, :tpp, :offReb, :defReb, :totReb, :assists, :pFouls,
                :steals, :turnovers, :blocks, :plusMinus
            );
        ''', player_data)
    conn.commit()

def main():
    seasons = [2022, 2023]
    for season in seasons:
        print(f"Processing season {season}...")
        create_games_table()
        create_player_game_stats_table(season)

        games = fetch_games(season)
        insert_games(games, season)

        for game in games:
            game_id = game["id"]
            players = fetch_player_stats(game_id)
            if players:
                insert_player_stats(players, season, game_id)
            time.sleep(DELAY)

    conn.close()

if __name__ == "__main__":
    main()
