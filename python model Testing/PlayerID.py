import sqlite3


conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()

def create_player_ids_table(season):
    table_name = f"player_ids_{season}"
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            Name TEXT,
            id1 TEXT,
            id2 TEXT,
            PRIMARY KEY (id1, id2)
        );
    ''')
    conn.commit()
    print(f"Table '{table_name}' created successfully.")

def populate_player_ids_table(season):
    players_table = f"Players_{season}"
    stats_table = f"PlayerGameStats_{season}"
    ids_table = f"player_ids_{season}"
    
    cursor.execute(f'''
        INSERT {ids_table} (Name, id1, id2)
        SELECT 
            p.name AS Name,
            p.player_id AS id1,
            s.player_id AS id2
        FROM {players_table} p
        INNER JOIN {stats_table} s
        ON p.name = (s.firstname || ' ' || s.lastname);
    ''')
    conn.commit()
    print(f"Data populated into '{ids_table}' for the {season} season successfully.")

def main():
    for season in [2022, 2023]:
        create_player_ids_table(season)
        populate_player_ids_table(season)
    conn.close()

if __name__ == "__main__":
    main()
