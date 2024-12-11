import sqlite3
import requests
import time


API_KEY = 'R3DS7uoSzMcPPBIG51BrTAv5gU4xMHcDHCwZNoyE'
API_URL = 'https://api.sportradar.com/nba/trial/v8/en/seasons/{year}/REG/teams/{team_id}/statistics.json'
DELAY = 2.5


conn = sqlite3.connect('nba_data.db')
cursor = conn.cursor()

def create_players_table(year):
    table_name = f"Players_{year}"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            player_id TEXT PRIMARY KEY,
            name TEXT,
            position TEXT,
            games_played INTEGER,
            games_started INTEGER,
            points REAL,
            rebounds REAL,
            assists REAL,
            steals REAL,
            blocks REAL,
            minutes REAL,
            turnovers REAL,
            personal_fouls REAL,
            field_goals_made REAL,
            field_goals_att REAL,
            field_goals_pct REAL,
            three_points_made REAL,
            three_points_att REAL,
            three_points_pct REAL,
            free_throws_made REAL,
            free_throws_att REAL,
            free_throws_pct REAL,
            double_doubles INTEGER,
            triple_doubles INTEGER,
            effective_fg_pct REAL,
            true_shooting_pct REAL,
            usage_rate REAL
        );
    ''')
    conn.commit()
    print(f"Table {table_name}")

def combine_player_stats(existing_stats, new_stats):

    combined_stats = existing_stats.copy()
    combined_stats['games_played'] += new_stats['games_played']
    combined_stats['games_started'] += new_stats['games_started']


    total_games = combined_stats['games_played']
    existing_games = existing_stats['games_played']
    new_games = new_stats['games_played']


    if combined_stats['field_goals_att'] > 0:
        combined_stats['field_goals_pct'] = round(
            combined_stats['field_goals_made'] / combined_stats['field_goals_att'], 2
        )
    if combined_stats['three_points_att'] > 0:
        combined_stats['three_points_pct'] = round(
            combined_stats['three_points_made'] / combined_stats['three_points_att'], 2
        )
    if combined_stats['free_throws_att'] > 0:
        combined_stats['free_throws_pct'] = round(
            combined_stats['free_throws_made'] / combined_stats['free_throws_att'], 2
        )

    # Correct Effective Field Goal Percentage
    if combined_stats['field_goals_att'] > 0:
        combined_stats['effective_fg_pct'] = round(
            (combined_stats['field_goals_made'] + 0.5 * combined_stats['three_points_made']) /
            combined_stats['field_goals_att'], 2
        )

    # For averages or per-game stats, calculate weighted averages
    if total_games > 0:
        for avg_key in [
            'points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes', 'turnovers',
            'personal_fouls' , 'field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 
            'free_throws_made', 'free_throws_att'
        ]:
            combined_stats[avg_key] = round(
                (existing_stats[avg_key] * existing_games +
                 new_stats[avg_key] * new_games)
                / total_games, 2
            )

    return combined_stats

def parse_player_stats(player):
    """
    Parse the player stats from the JSON response.
    """
    stats = {}
    totals = player.get("total", {})
    averages = player.get("average", {})

    stats.update({
        'player_id': player.get('id', 'N/A'),
        'name': player.get('full_name', 'Unknown'),
        'position': player.get('position', 'Unknown'),
        'games_played': totals.get('games_played', 0),
        'games_started': totals.get('games_started', 0),
        'points': averages.get('points', 0),
        'rebounds': averages.get('rebounds', 0),
        'assists': averages.get('assists', 0),
        'steals': averages.get('steals', 0),
        'blocks': averages.get('blocks', 0),
        'minutes': averages.get('minutes', 0),
        'turnovers': averages.get('turnovers', 0),
        'personal_fouls': averages.get('personal_fouls', 0),
        'field_goals_made': averages.get('field_goals_made', 0),
        'field_goals_att': averages.get('field_goals_att', 0),
        'field_goals_pct': totals.get('field_goals_pct', 0),
        'three_points_pct': totals.get('three_points_pct', 0),
        'free_throws_pct': totals.get('free_throws_pct', 0),
        'three_points_made': averages.get('three_points_made', 0),
        'three_points_att': averages.get('three_points_att', 0),
        'free_throws_made': averages.get('free_throws_made', 0),
        'free_throws_att': averages.get('free_throws_att', 0),
        'double_doubles': totals.get('double_doubles', 0),
        'triple_doubles': totals.get('triple_doubles', 0),
        'usage_rate': totals.get('usage_pct', 0),
        'effective_fg_pct': totals.get('effective_fg_pct', 0),
        'true_shooting_pct': totals.get('true_shooting_pct', 0)
    })
    return stats

def fetch_team_stats(year, team_id, player_data):
    """
    Fetch team stats from the API and combine data for players across multiple teams.
    """
    url = API_URL.format(year=year, team_id=team_id)
    try:
        response = requests.get(url, params={"api_key": API_KEY})
        response.raise_for_status()
        team_data = response.json()

        players = team_data.get('players', [])
        for player in players:
            player_id = player.get('id')
            player_stats = parse_player_stats(player)

            if player_id in player_data:
                player_data[player_id] = combine_player_stats(player_data[player_id], player_stats)
            else:
                player_data[player_id] = player_stats

    except requests.exceptions.RequestException as e:
        print(f"Error fetching stats for team {team_id} in {year}: {e}")

def insert_player_data(year, player_stats):
    """
    Insert a player's stats into the database.
    """
    table_name = f"Players_{year}"

    cursor.execute(f'''
        INSERT OR REPLACE INTO {table_name} (
            player_id, name, position, games_played, games_started,
            points, rebounds, assists, steals, blocks, minutes, turnovers,
            personal_fouls, field_goals_made, field_goals_att, field_goals_pct,
            three_points_made, three_points_att, three_points_pct,
            free_throws_made, free_throws_att, free_throws_pct,
            double_doubles, triple_doubles, effective_fg_pct,
            true_shooting_pct, usage_rate
        ) VALUES (
            :player_id, :name, :position, :games_played, :games_started,
            :points, :rebounds, :assists, :steals, :blocks, :minutes, :turnovers,
            :personal_fouls, :field_goals_made, :field_goals_att, :field_goals_pct,
            :three_points_made, :three_points_att, :three_points_pct,
            :free_throws_made, :free_throws_att, :free_throws_pct,
            :double_doubles, :triple_doubles, :effective_fg_pct,
            :true_shooting_pct, :usage_rate
        );
    ''', player_stats)
    conn.commit()
    print(f"Inserted data for player {player_stats['name']} into {table_name}.")

def get_team_ids():
    """
    Fetch team IDs from the database.
    """
    cursor.execute("SELECT id FROM Teams")
    return [row[0] for row in cursor.fetchall()]

def main():
    """
    Main function to orchestrate fetching and storing stats for all teams and years.
    """
    years = [2022, 2023]
    team_ids = get_team_ids()
    if not team_ids:
        print("No team IDs found in the database. Exiting.")
        return

    for year in years:
        print(f"Processing stats for the {year} season...")
        create_players_table(year)
        player_data = {}

        for team_id in team_ids:
            fetch_team_stats(year, team_id, player_data)
            time.sleep(DELAY)

        for stats in player_data.values():
            insert_player_data(year, stats)

    conn.close()

if __name__ == "__main__":
    main()
