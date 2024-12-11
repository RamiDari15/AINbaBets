import sqlite3
import requests
import time


API_KEY = 'R3DS7uoSzMcPPBIG51BrTAv5gU4xMHcDHCwZNoyE'
TEAM_STATS_API = 'https://api.sportradar.com/nba/trial/v8/en/seasons/{year}/REG/teams/{team_id}/statistics.json'
DELAY = 2.5

conn = sqlite3.connect('nba_data.db')
cursor = conn.cursor()

def create_combined_table(year):
    table_name = f"TeamStats_{year}"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            team_id TEXT PRIMARY KEY,
            name TEXT,
            market TEXT,
            fast_break_pts_own REAL,
            fast_break_pts_opp REAL,
            points_off_turnovers_own REAL,
            points_off_turnovers_opp REAL,
            minutes_own REAL,
            minutes_opp REAL,
            points_own REAL,
            points_opp REAL,
            off_rebounds_own REAL,
            off_rebounds_opp REAL,
            def_rebounds_own REAL,
            def_rebounds_opp REAL,
            rebounds_own REAL,
            rebounds_opp REAL,
            assists_own REAL,
            assists_opp REAL,
            steals_own REAL,
            steals_opp REAL,
            blocks_own REAL,
            blocks_opp REAL,
            turnovers_own REAL,
            turnovers_opp REAL,
            field_goals_made_own REAL,
            field_goals_made_opp REAL,
            bench_points_own REAL,
            bench_points_opp REAL
        );
    ''')
    conn.commit()
    print(f"Table {table_name}")

def parse_combined_stats(team_data):
    own_stats = team_data['own_record']['average']
    opponent_stats = team_data['opponents']['average']

    return {
        'team_id': team_data['id'],
        'name': team_data['name'],
        'market': team_data['market'],
        'fast_break_pts_own': own_stats.get('fast_break_pts', 0),
        'fast_break_pts_opp': opponent_stats.get('fast_break_pts', 0),
        'points_off_turnovers_own': own_stats.get('points_off_turnovers', 0),
        'points_off_turnovers_opp': opponent_stats.get('points_off_turnovers', 0),
        'minutes_own': own_stats.get('minutes', 0),
        'minutes_opp': opponent_stats.get('minutes', 0),
        'points_own': own_stats.get('points', 0),
        'points_opp': opponent_stats.get('points', 0),
        'off_rebounds_own': own_stats.get('off_rebounds', 0),
        'off_rebounds_opp': opponent_stats.get('off_rebounds', 0),
        'def_rebounds_own': own_stats.get('def_rebounds', 0),
        'def_rebounds_opp': opponent_stats.get('def_rebounds', 0),
        'rebounds_own': own_stats.get('rebounds', 0),
        'rebounds_opp': opponent_stats.get('rebounds', 0),
        'assists_own': own_stats.get('assists', 0),
        'assists_opp': opponent_stats.get('assists', 0),
        'steals_own': own_stats.get('steals', 0),
        'steals_opp': opponent_stats.get('steals', 0),
        'blocks_own': own_stats.get('blocks', 0),
        'blocks_opp': opponent_stats.get('blocks', 0),
        'turnovers_own': own_stats.get('turnovers', 0),
        'turnovers_opp': opponent_stats.get('turnovers', 0),
        'field_goals_made_own': own_stats.get('field_goals_made', 0),
        'field_goals_made_opp': opponent_stats.get('field_goals_made', 0),
        'bench_points_own': own_stats.get('bench_points', 0),
        'bench_points_opp': opponent_stats.get('bench_points', 0)
    }

def fetch_team_stats(year, team_id):
    url = TEAM_STATS_API.format(year=year, team_id=team_id)

    response = requests.get(url, params={"api_key": API_KEY})
    response.raise_for_status()
    team_data = response.json()
    time.sleep(DELAY)  

    return parse_combined_stats(team_data)


def insert_combined_stats(year, combined_stats):
    table_name = f"TeamStats_{year}"
    cursor.execute(f'''
        INSERT OR REPLACE INTO {table_name} (
            team_id, name, market,
            fast_break_pts_own, fast_break_pts_opp,
            points_off_turnovers_own, points_off_turnovers_opp,
            minutes_own, minutes_opp,
            points_own, points_opp,
            off_rebounds_own, off_rebounds_opp,
            def_rebounds_own, def_rebounds_opp,
            rebounds_own, rebounds_opp,
            assists_own, assists_opp,
            steals_own, steals_opp,
            blocks_own, blocks_opp,
            turnovers_own, turnovers_opp,
            field_goals_made_own, field_goals_made_opp,
            bench_points_own, bench_points_opp
        ) VALUES (
            :team_id, :name, :market,
            :fast_break_pts_own, :fast_break_pts_opp,
            :points_off_turnovers_own, :points_off_turnovers_opp,
            :minutes_own, :minutes_opp,
            :points_own, :points_opp,
            :off_rebounds_own, :off_rebounds_opp,
            :def_rebounds_own, :def_rebounds_opp,
            :rebounds_own, :rebounds_opp,
            :assists_own, :assists_opp,
            :steals_own, :steals_opp,
            :blocks_own, :blocks_opp,
            :turnovers_own, :turnovers_opp,
            :field_goals_made_own, :field_goals_made_opp,
            :bench_points_own, :bench_points_opp
        );
    ''', combined_stats)
    conn.commit()

def get_team_ids():
    cursor.execute("SELECT id FROM Teams")
    return [row[0] for row in cursor.fetchall()]

def main():
    years = [2022, 2023]
    team_ids = get_team_ids()

    for year in years:
        print(f"Processing stats for the {year} season...")
        create_combined_table(year)

        for team_id in team_ids:
            combined_stats = fetch_team_stats(year, team_id)
            if combined_stats:
                insert_combined_stats(year, combined_stats)

    conn.close()

if __name__ == "__main__":
    main()
