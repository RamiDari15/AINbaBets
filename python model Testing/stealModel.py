import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()

query = """
SELECT * FROM (
    -- 2022 Data
    SELECT 
        pgs.steals AS actual_steals,
        p.games_played,
        p.steals AS avg_steals,
        p.minutes AS avg_minutes,
        ts_own.steals_own AS player_team_steals_own,
        ts_own.turnovers_opp AS player_team_turnovers_opp,
        ts_opp.turnovers_own AS opponent_team_turnovers_own
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
        pgs.steals AS actual_steals,
        p.games_played,
        p.steals AS avg_steals,
        p.minutes AS avg_minutes,
        ts_own.steals_own AS player_team_steals_own,
        ts_own.turnovers_opp AS player_team_turnovers_opp,
        ts_opp.turnovers_own AS opponent_team_turnovers_own
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
    "actual_steals",
    "games_played", "avg_steals", "avg_minutes",
    "player_team_steals_own", "player_team_turnovers_opp",
    "opponent_team_turnovers_own"
]

all_steal_data = []
for row in results:
    actual_steals = row[0]
    data = row[1:]

    for steal_line in range(0, 4):  # Predict 0–3 steals
        binary_output = 1 if actual_steals > steal_line else 0
        entry = list(data) + [steal_line, binary_output]
        all_steal_data.append(entry)

columns = columns[1:] + ["steal_line", "binary_output"]
df = pd.DataFrame(all_steal_data, columns=columns)
conn.close()

X = df.drop(columns=["binary_output"]).values
y = df["binary_output"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP Model for steals
class NBAStealModel(nn.Module):
    def __init__(self, input_dim):
        super(NBAStealModel, self).__init__()
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
model = NBAStealModel(input_size)
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 150
batch_size = 64

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

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

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_predictions = test_predictions.round()
    accuracy = (test_predictions.numpy() == y_test_tensor.numpy()).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
