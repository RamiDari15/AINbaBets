import requests
import sqlite3
import time


API_KEY = 'v7mGqb1klOMT8wHHK7Vp8uBXaemKmLZ9g2oc4ew5'
API_URL = 'https://api.sportradar.com/nba/trial/v8/en/league/hierarchy.json'

conn = sqlite3.connect('nba_data.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS Teams (
        id TEXT PRIMARY KEY,
        name TEXT,
        alias TEXT,
        market TEXT
    )
''')

def fetch_team_data():
    try:
        response = requests.get(API_URL, params={'api_key': API_KEY})
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def insert_team_data(team):
    cursor.execute('''
        INSERT OR REPLACE INTO Teams (id, name, alias, market)
        VALUES (?, ?, ?, ?)
    ''', (team['id'], team['name'], team['alias'], team['market']))
    conn.commit()

if __name__ == '__main__':
    data = fetch_team_data()
    if data:
        for conference in data.get('conferences', []):
            for division in conference.get('divisions', []):
                for team in division.get('teams', []):
                    insert_team_data(team)
                    print(f"Inserted team: {team['name']}")

    conn.close()
