import sqlite3
import pandas as pd


conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()

## Practice calling Correct query
query = """
  pgs.points AS actual_points,
        p.games_played,
        p.points AS avg_points,
        p.field_goals_made,
        p.field_goals_att,
        p.field_goals_pct,
        p.three_points_made,
        p.three_points_att,
        p.three_points_pct,
        p.free_throws_made,
        p.free_throws_att,
        p.free_throws_pct,
        p.true_shooting_pct,
        p.usage_rate,
        p.minutes AS avg_minutes,
        ts_own.fast_break_pts_own AS player_team_fast_break_pts,
        ts_own.points_off_turnovers_own AS player_team_points_off_turnovers,
        ts_own.points_own AS player_team_points,
        ts_own.rebounds_own AS player_team_rebounds,
        ts_own.turnovers_own AS player_team_turnovers,
        ts_own.turnovers_opp AS player_team_turnovers_opp,
        ts_opp.fast_break_pts_opp AS opponent_team_fast_break_pts,
        ts_opp.points_off_turnovers_opp AS opponent_team_points_off_turnovers,
        ts_opp.points_opp AS opponent_team_points,
        ts_opp.turnovers_own AS opponent_team_turnovers,
        ts_opp.turnovers_opp AS opponent_team_turnovers_opp
    FROM PlayerGameStats_2022 AS pgs
    JOIN player_ids_2022 AS pid ON pgs.player_id = pid.id2
    JOIN Players_2022 AS p ON pid.id1 = p.player_id
    JOIN UnifiedTeamIds AS tu_player ON pgs.team_id = tu_player.game_team_id
    JOIN Game AS g ON g.game_id = pgs.game_id
    JOIN UnifiedTeamIds AS tu_opponent ON tu_opponent.game_team_id = 
        CASE
            WHEN g.home_team_id = pgs.team_id THEN g.visitors_team_id
            WHEN g.visitors_team_id = pgs.team_id THEN g.home_team_id
            ELSE NULL
        END
    JOIN TeamStats_2022 AS ts_own ON tu_player.team_id = ts_own.team_id
    JOIN TeamStats_2022 AS ts_opp ON tu_opponent.team_id = ts_opp.team_id
    UNION ALL
    -- 2023 Data
    SELECT 
        pgs.points AS actual_points,
        p.games_played,
        p.points AS avg_points,
        p.field_goals_made,
        p.field_goals_att,
        p.field_goals_pct,
        p.three_points_made,
        p.three_points_att,
        p.three_points_pct,
        p.free_throws_made,
        p.free_throws_att,
        p.free_throws_pct,
        p.true_shooting_pct,
        p.usage_rate,
        p.minutes AS avg_minutes,
        ts_own.fast_break_pts_own AS player_team_fast_break_pts,
        ts_own.points_off_turnovers_own AS player_team_points_off_turnovers,
        ts_own.points_own AS player_team_points,
        ts_own.rebounds_own AS player_team_rebounds,
        ts_own.turnovers_own AS player_team_turnovers,
        ts_own.turnovers_opp AS player_team_turnovers_opp,
        ts_opp.fast_break_pts_opp AS opponent_team_fast_break_pts,
        ts_opp.points_off_turnovers_opp AS opponent_team_points_off_turnovers,
        ts_opp.points_opp AS opponent_team_points,
        ts_opp.turnovers_own AS opponent_team_turnovers,
        ts_opp.turnovers_opp AS opponent_team_turnovers_opp
    FROM PlayerGameStats_2023 AS pgs
    JOIN player_ids_2023 AS pid ON pgs.player_id = pid.id2
    JOIN Players_2023 AS p ON pid.id1 = p.player_id
    JOIN UnifiedTeamIds AS tu_player ON pgs.team_id = tu_player.game_team_id
    JOIN Game AS g ON g.game_id = pgs.game_id
    JOIN UnifiedTeamIds AS tu_opponent ON tu_opponent.game_team_id = 
        CASE
            WHEN g.home_team_id = pgs.team_id THEN g.visitors_team_id
            WHEN g.visitors_team_id = pgs.team_id THEN g.home_team_id
            ELSE NULL
        END
    JOIN TeamStats_2023 AS ts_own ON tu_player.team_id = ts_own.team_id
    JOIN TeamStats_2023 AS ts_opp ON tu_opponent.team_id = ts_opp.team_id
) AS combined_data;

LIMIT 1;
"""



# Create Columns names
columns = [
    "player_id", "game_id", "player_name", "actual_points",
    "games_played", "avg_points", "field_goals_made", "field_goals_att", "field_goals_pct",
    "three_points_made", "three_points_att", "three_points_pct", "free_throws_made",
    "free_throws_att", "free_throws_pct", "true_shooting_pct", "usage_rate", "avg_minutes",
    "player_team_fast_break_pts", "player_team_points_off_turnovers", "player_team_points",
    "player_team_rebounds", "player_team_turnovers", "player_team_turnovers_opp",
    "opponent_team_fast_break_pts", "opponent_team_points_off_turnovers", "opponent_team_points",
    "opponent_team_turnovers", "opponent_team_turnovers_opp"
]

cursor.execute(query)
result = cursor.fetchone()

df = pd.DataFrame([result], columns=columns)
df.to_csv("player_team_opponent_stats.csv", index=False)

conn.close()
