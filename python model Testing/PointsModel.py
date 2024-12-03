#changed everything to torch from manual
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlite3
import pandas as pd



conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()


#Grab 27 input features
query = """
SELECT * FROM (
    -- 2022 Data
    SELECT 
        pgs.points AS actual_points,
        pgs.minutes AS minutes,
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
        pgs.minutes AS minutes,
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


"""


cursor.execute(query)
results = cursor.fetchall()


columns = [
    "actual_points",
    "games_played", "avg_points", "field_goals_made", "field_goals_att", "field_goals_pct",
    "three_points_made", "three_points_att", "three_points_pct", "free_throws_made",
    "free_throws_att", "free_throws_pct", "true_shooting_pct", "usage_rate", "avg_minutes",
    "player_team_fast_break_pts", "player_team_points_off_turnovers", "player_team_points",
    "player_team_rebounds", "player_team_turnovers", "player_team_turnovers_opp",
    "opponent_team_fast_break_pts", "opponent_team_points_off_turnovers", "opponent_team_points",
    "opponent_team_turnovers", "opponent_team_turnovers_opp"
]


all_box_scores = []


# Process each result to generate 35 entries per statbox
for row in results:
    actual_points = row[0]  
    data = row[1:]     
    


    for point_line in range(1, 36):  
        binary_output = 1 if actual_points > point_line else 0
        entry = list(data) + [point_line, binary_output]  
        
        all_box_scores.append(entry) 

stats = columns[1:] + ["point_line", "binary_output"]
df = pd.DataFrame(all_box_scores, columns=stats)
conn.close()

# Separate features and target
X = df.drop(columns=["binary_output"]).values
y = df["binary_output"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Create MLP model to help lose money
class NBAPointModel(nn.Module):
    def __init__(self, input_dim):
        super(NBAPointModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x




input_size = X_train.shape[1]
model = NBAPointModel(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 150
batch_size = 64

#Chat helped  explain tensor and torch util 
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
best_accuracy = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_predictions = test_predictions.round()
        accuracy = (test_predictions.numpy() == y_test_tensor.numpy()).mean()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy * 100:.2f}%")


    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict() 
        torch.save(best_model_state, 'best_points_model1.pt')  
        print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")



best_model = NBAPointModel(input_size)
best_model.load_state_dict(torch.load('best_points_model1.pt'))
best_model.eval()

with torch.no_grad():
    test_predictions = best_model(X_test_tensor)
    test_predictions = test_predictions.round()
    accuracy = (test_predictions.numpy() == y_test_tensor.numpy()).mean()
    print(f"Test Accuracy of Best Model: {accuracy * 100:.2f}%")