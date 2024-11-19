import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlite3
import pandas as pd

conn = sqlite3.connect("nba_data.db")
cursor = conn.cursor()


query = """
SELECT * FROM (
    -- 2022 Data
    SELECT 
        pgs.assists AS actual_assists,
        p.games_played,
        p.assists AS avg_assists,
        p.turnovers,
        p.usage_rate,
        p.minutes AS avg_minutes,
        ts_own.fast_break_pts_own AS player_team_fast_break_pts,
        ts_own.assists_own AS player_team_assists,
        ts_opp.assists_opp AS opponent_team_assists
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
        pgs.assists AS actual_assists,
        p.games_played,
        p.assists AS avg_assists,
        p.turnovers,
        p.usage_rate,
        p.minutes AS avg_minutes,
        ts_own.fast_break_pts_own AS player_team_fast_break_pts,
        ts_own.assists_own AS player_team_assists,
        ts_opp.assists_opp AS opponent_team_assists
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
    "actual_assists",
    "games_played", "avg_assists", "turnovers", "usage_rate", "avg_minutes",
    "player_team_fast_break_pts", "player_team_assists", "opponent_team_assists"
]


all_box_scores = []

for row in results:
    actual_assists = row[0]
    data = row[1:]

    for assist_line in range(1, 11): 
        binary_output = 1 if actual_assists > assist_line else 0
        entry = list(data) + [assist_line, binary_output]
        all_box_scores.append(entry)

stats = columns[1:] + ["assist_line", "binary_output"]
df = pd.DataFrame(all_box_scores, columns=stats)
conn.close()

X = df.drop(columns=["binary_output"]).values
y = df["binary_output"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#MLP model for assists
class NBAAssistModel(nn.Module):
    def __init__(self, input_dim):
        super(NBAAssistModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  
        self.fc4 = nn.Linear(32, 16)  
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))  
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc5(x))
        return x

# Initialize the model
input_size = X_train.shape[1]
model = NBAAssistModel(input_size)
criterion = nn.BCELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training the model
epochs = 300
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
