import sqlite3

# Database connection
conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()

def create_unified_team_ids_table():
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UnifiedTeamIds (
            team_name TEXT PRIMARY KEY,
            market TEXT,
            team_id TEXT,
            game_team_id TEXT
        );
    ''')
    conn.commit()
    print("UnifiedTeamIds table")

def populate_unified_team_ids():

    cursor.execute('''
        SELECT DISTINCT t.name || ' ' || t.market AS team_name, t.id AS team_id
        FROM Teams t
    ''')
    teams_data = cursor.fetchall()

    game_team_ids = {}
    cursor.execute('''
        SELECT DISTINCT g.home_team_name, g.home_team_id
        FROM Game g
        UNION
        SELECT DISTINCT g.visitors_team_name, g.visitors_team_id
        FROM Game g
    ''')
    for row in cursor.fetchall():
        game_team_ids[row[0]] = row[1] 

    for team_name, team_id in teams_data:
        game_team_id = game_team_ids.get(team_name, None)  
        cursor.execute('''
            INSERT OR REPLACE INTO UnifiedTeamIds (team_name, market, team_id, game_team_id)
            VALUES (?, ?, ?, ?)
        ''', (team_name, team_name.split()[-1], team_id, game_team_id))  
    conn.commit()


def main():
    create_unified_team_ids_table()
    populate_unified_team_ids()
    conn.close()

if __name__ == "__main__":
    main()
