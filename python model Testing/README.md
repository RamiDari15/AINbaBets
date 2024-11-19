README
Project Overview
This project collects, processes, and analyzes NBA player and team data to predict player performance in games, specifically if a player will score over a certain point threshold. Data is fetched from the Sportradar API and stored in a SQLite database, then used to train a machine learning model for predictions.

Points Model Description
        Points model tries to predict if a player will score over a specific points between(e.g., 0-35 points) 
    From PlayerGameStats Table:
        points (points scored in the game)

    From Players Table:
    games_played
    points (average points per game)
    field_goals_made
    field_goals_att
    field_goals_pct
    three_points_made
    three_points_att
    three_points_pct
    free_throws_made
    free_throws_att
    free_throws_pct
    true_shooting_pct
    usage_rate
    minutes (average minutes per game)
From TeamStats Table:

    fast_break_pts_own (playes team)
    points_off_turnovers_own ( players team)
    points_own (player's team)
    rebounds_own ( players team)
    turnovers_own (player's team)
    turnovers_opp (opponet team)
    fast_break_pts_opp (oppnent team)
    points_off_turnovers_opp (oppnent team)
    points_opp ( oppnent team)


Assist model trys to predict if someone will get 0-10 assist, use assists in PlayerGameStats, 
    From PlayerGameStats Table:
        assists (assists in the game)
    From Players Table:
        games_played
        assists
        minutes
        usage_rate
        turnovers
    From TeamStats Table:
        fast_break (from player team)
        assists_own (from player team)
        assists_opp (from opponent team)



rebound model trys to predict if someone will get 0-15 reboundss, us reboundin PlayerGameStats
    From PlayerGameStats Table:
        totReb (rebounds in the game)
        From Players Table:
        games_played
        rebounds
        blocks

    From TeamStats Table:
        off_rebounds_own (from player team)
        off_rebound_opp (fromm player team)
        def_rebounds_own (from player team)
        def_rebound_opp (fromm player team)
        rebounds_opp (from player team)
        rebounds_own (from player team)
        field_goal_made_opp (from player team)
        blocks_own (from player team)
        off_rebounds_own (from opponet team)
        off_rebound_opp (fromm opponet team)
        def_rebounds_own (from opponet team)
        def_rebound_opp (fromm opponet team)
        rebounds_own (from opponet team)
        rebounds_opp (from opponet team)
        field_goal_made_ own (from opponet team)
        blocks_opp (from opponet team)


block model trys to predict if someone will get 0-3 blocks, use blocks in PlayerGameStats
    From PlayerGameStats Table:
        blocks
        From Players Table:
        games_played
        rebounds
        blocks
        minutes

    From TeamStats Table:
        off_rebound_opp (fromm player team)
        rebounds_opp (from player team)
        rebounds_own (from player team)
        field_goal_made_opp (from player team)
        blocks_own (from player team)
        off_rebounds_own (from opponet team)
        blocks_opp (from opponet team)


steals model trys to predict if someone will get 0-3 blocks instead of 0-35 points, use steals in PlayerGameStats
    From PlayerGameStats Table:
        steals
        From Players Table:
        games_played
        steals
        minutes

    From TeamStats Table:
        steals_own(fromm player team)
        turnovers_opp(from player team)
        turnovers_own(from opponet team

Files

PointsModel.py
Builds a MLP to predict if a player will score over a certain point threshold in a game.
Uses player, team, and game stats as inputs to make predictions.
Includes adjustable parameters for better model performance.

team_stats.py
Gathers team-level stats like points, rebounds, turnovers, and advanced metrics.
Stores stats in yearly tables (2022 and 2023).

team.py
Pulls basic details for each NBA team, like team name, market, and unique IDs.
Fills the Teams table to be used as a reference.

team_ref.py
Links team data across multiple tables using unified IDs.
Simplifies queries and ensures consistency.

stats.py
Fetches player stats for 2022 and 2023 seasons.
Handles cases where players switch teams by combining their stats properly.
Tracks advanced metrics like shooting percentages, usage rates, and per-game averages.

seasons.py
Manages season data for better organization and analysis.
Helps focus on season-specific stats like 2022 or 2023.

SampleInputPtsData.py
Prepares data for the machine learning model.
Combines player averages, team stats, and game performance into a single dataset.

gamesStats.py
Pulls player stats for individual games, like points, assists, and rebounds.
Creates tables (PlayerGameStats_2022, PlayerGameStats_2023) to store game-by-game stats for each player.

games.py
Fetches schedules for 2022 and 2023 seasons.
Captures game details like scores, home/away teams, and venues.
Links games to other tables using game_id.

PlayerID.py
Resolves differences in player IDs across data sources.
Maps multiple IDs to a unified player_ids table.


1. Teams
    Description:  
        Stores basic details for NBA teams.

    Columns:
        team_id: Unique identifier for each team (primary key).
        name: Name of the team.
        market: Market or city of the team.
        sr_id: Sportradar ID for reference.
        reference: External ID for cross-referencing.

    Connections:
        Linked to TeamStats and games using team_id.


2. TeamStats (2022 and 2023)
    Description: 
        Stores team performance metrics for each season.

    Columns:
        team_id: Reference to the Teams table.
        fast_break_pts, points_off_turnovers, points, rebounds: Performance metrics.
        offensive_rating, defensive_rating, net_rating: Advanced metrics.
        Separate columns for team and opponent stats (e.g., points_own, points_opp).

    Connections:
        Linked to Teams via team_id.
        Used in queries to enrich game and player stats.

    3. Games
        Description: Stores schedule and basic details for NBA games.

    Columns:
        game_id: Unique identifier for each game (primary key).
        date: Date of the game.
        home_team_id: ID of the home team (foreign key to Teams).
        visitors_team_id: ID of the visiting team (foreign key to Teams).
        score_home, score_visitors: Final scores for the game.

    Connections:

        Linked to Teams via home_team_id and visitors_team_id.
        Acts as a reference for player and team game stats.

4. Players (2022 and 2023)
    Description: 
        stores individual player stats for each season.

    Columns:
        player_id: Unique identifier for each player (primary key).
        name, position: Player details.
        games_played, avg_points, field_goals_made, etc.: Season-level averages and totals.
        Advanced metrics: usage_rate, true_shooting_pct.
    Connections:
        Linked to PlayerGameStats and PlayerID.
        Unified across seasons using PlayerID.
5. PlayerGameStats (2022 and 2023)
    Description: 
        Stores detailed stats for individual players in each game.

    Columns:
        game_id: Reference to the Games table.
        player_id: Reference to the Players table.
        team_id: Reference to the Teams table.
        points, rebounds, assists, etc.: Game-level performance metrics.
        Advanced metrics: efficiency, offensive_rating, defensive_rating.
    Connections:
        Linked to Games via game_id.
        Linked to Players via player_id.
        Linked to Teams via team_id.
6. PlayerID
    Description:
         Resolves discrepancies in player IDs across data sources.

`   Columns:
        id1: Primary player ID from Players.
        id2: Alternate ID used in PlayerGameStats.
        name: player name
    Connections:
       Links Players and PlayerGameStats by mapping their IDs.

7. UnifiedTeamIds
    Description:
         Resolves discrepancies in team IDs across data sources.

    Columns:
        team_id: Primary team ID from Teams.
        game_team_id: Alternate ID used in game and stats data.

    Connections:
        Links Teams to Games, TeamStats, and other tables.
        Connections Summary
        Teams - TeamStats: Links team performance stats for each season.
        Teams - Games: Links home and visiting teams to game schedules.
        Games - PlayerGameStats: Links player performance data to specific games.
        Players - PlayerGameStats: Links individual game performance to player profiles.
        PlayerID - Players - PlayerGameStats: Resolves ID discrepancies across tables and seasons.
        UnifiedTeamIds - Teams - Games/TeamStats: Ensures consistency across team IDs in different datasets.
