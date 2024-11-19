import requests
import sqlite3


API_KEY = 'v7mGqb1klOMT8wHHK7Vp8uBXaemKmLZ9g2oc4ew5'
API_URL = 'https://api.sportradar.com/nba/trial/v8/en/league/seasons.json'


conn = sqlite3.connect('nba_data.db')
cursor = conn.cursor()


cursor.execute('''
    CREATE TABLE Seasons (
        season_id TEXT PRIMARY KEY,
        year INTEGER,
        type TEXT
    )
''')


def fetch_season_data():
    try:
        response = requests.get(API_URL, params={'api_key': API_KEY})
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def insert_season_data(season):
    season_id = season.get('id')  
    year = season.get('year')
    season_type = season.get('type')  

    print(f"Inserting into database: season_id={season_id}, year={year}, type={season_type}")


    cursor.execute('''
        INSERT OR REPLACE INTO Seasons (season_id, year, type)
        VALUES (?, ?, ?)
    ''', (season_id, year, season_type))
    conn.commit()

# Main script execution
if __name__ == '__main__':
    data = fetch_season_data()
    if data and 'seasons' in data:
        for season in data['seasons']:
            if season.get('type') == 'REG' and season.get('year') in [2022, 2023]:
                insert_season_data(season)
                print(f"Inserted season: {season['year']} ({season['type']})")
    
    conn.close()
