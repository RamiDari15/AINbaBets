import requests
import sqlite3


API_KEY = 'kyGjXndtvJPZrqlHH0IG1SBthNkHp6OomlT3s20H'
API_URL = 'https://api.sportradar.com/nba/trial/v8/en/games/{year}/REG/schedule.json'


conn = sqlite3.connect('nba_data.db')
cursor = conn.cursor()

cursor.execute('''
    DROP TABLE IF EXISTS Games;
''')
cursor.execute('''
    CREATE TABLE IF NOT EXISTS Games (
        game_id TEXT PRIMARY KEY,
        year INTEGER NOT NULL,
        home_team_id TEXT NOT NULL,
        away_team_id TEXT NOT NULL,
        scheduled TEXT,
        venue TEXT,
        status TEXT,
        home_score INTEGER,
        away_score INTEGER
    );
''')

def fetch_games_data(year):
    url = API_URL.format(year=year)
    response = requests.get(url, params={'api_key': API_KEY})
    response.raise_for_status()  
    return response.json()



def insert_game_data(game, year):
    game_id = game.get('id')
    home_team_id = game.get('home', {}).get('id')
    away_team_id = game.get('away', {}).get('id')
    scheduled = game.get('scheduled')
    venue = game.get('venue', {}).get('name') if game.get('venue') else None
    status = game.get('status')
    home_score = game.get('home_points', None)
    away_score = game.get('away_points', None)


    print(f"Inserting game: {game_id}, year: {year}, home: {home_team_id}, away: {away_team_id}")

    cursor.execute('''
        INSERT INTO Games (
            game_id, year, home_team_id, away_team_id, scheduled, venue, status, home_score, away_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (game_id, year, home_team_id, away_team_id, scheduled, venue, status, home_score, away_score))
    conn.commit()


if __name__ == '__main__':
    years = [2022, 2023]

    for year in years:
        data = fetch_games_data(year)
        if data and 'games' in data:
            for game in data['games']:
                insert_game_data(game, year)


    conn.close()
