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
        pgs.totReb AS actual_rebounds,
        p.games_played,
        p.rebounds AS avg_rebounds,
        p.blocks AS player_blocks,
        ts_own.off_rebounds_own AS player_team_off_rebounds_own,
        ts_own.off_rebounds_opp AS player_team_off_rebounds_opp,
        ts_own.def_rebounds_own AS player_team_def_rebounds_own,
        ts_own.def_rebounds_opp AS player_team_def_rebounds_opp,
        ts_own.rebounds_own AS player_team_rebounds_own,
        ts_own.rebounds_opp AS player_team_rebounds_opp,
        ts_own.field_goals_made_opp AS player_team_field_goals_made_opp,
        ts_own.blocks_own AS player_team_blocks_own,
        ts_opp.off_rebounds_own AS opponent_team_off_rebounds_own,
        ts_opp.off_rebounds_opp AS opponent_team_off_rebounds_opp,
        ts_opp.def_rebounds_own AS opponent_team_def_rebounds_own,
        ts_opp.def_rebounds_opp AS opponent_team_def_rebounds_opp,
        ts_opp.rebounds_own AS opponent_team_rebounds_own,
        ts_opp.rebounds_opp AS opponent_team_rebounds_opp,
        ts_opp.field_goals_made_own AS opponent_team_field_goals_made_own,
        ts_opp.blocks_opp AS opponent_team_blocks_opp
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
        pgs.totReb AS actual_rebounds,
        p.games_played,
        p.rebounds AS avg_rebounds,
        p.blocks AS player_blocks,
        ts_own.off_rebounds_own AS player_team_off_rebounds_own,
        ts_own.off_rebounds_opp AS player_team_off_rebounds_opp,
        ts_own.def_rebounds_own AS player_team_def_rebounds_own,
        ts_own.def_rebounds_opp AS player_team_def_rebounds_opp,
        ts_own.rebounds_own AS player_team_rebounds_own,
        ts_own.rebounds_opp AS player_team_rebounds_opp,
        ts_own.field_goals_made_opp AS player_team_field_goals_made_opp,
        ts_own.blocks_own AS player_team_blocks_own,
        ts_opp.off_rebounds_own AS opponent_team_off_rebounds_own,
        ts_opp.off_rebounds_opp AS opponent_team_off_rebounds_opp,
        ts_opp.def_rebounds_own AS opponent_team_def_rebounds_own,
        ts_opp.def_rebounds_opp AS opponent_team_def_rebounds_opp,
        ts_opp.rebounds_own AS opponent_team_rebounds_own,
        ts_opp.rebounds_opp AS opponent_team_rebounds_opp,
        ts_opp.field_goals_made_own AS opponent_team_field_goals_made_own,
        ts_opp.blocks_opp AS opponent_team_blocks_opp
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
    "actual_rebounds", "games_played", "avg_rebounds", "player_blocks",
    "player_team_off_rebounds_own", "player_team_off_rebounds_opp",
    "player_team_def_rebounds_own", "player_team_def_rebounds_opp",
    "player_team_rebounds_own", "player_team_rebounds_opp",
    "player_team_field_goals_made_opp", "player_team_blocks_own",
    "opponent_team_off_rebounds_own", "opponent_team_off_rebounds_opp",
    "opponent_team_def_rebounds_own", "opponent_team_def_rebounds_opp",
    "opponent_team_rebounds_own", "opponent_team_rebounds_opp",
    "opponent_team_field_goals_made_own", "opponent_team_blocks_opp"
]


all_box_scores = []
for row in results:
    actual_rebounds = row[0]
    data = row[1:]
    for rebound_line in range(1, 16):  # Generate predictions for rebound lines 1-15
        binary_output = 1 if actual_rebounds > rebound_line else 0
        entry = list(data) + [rebound_line, binary_output]
        all_box_scores.append(entry)

stats = columns[1:] + ["rebound_line", "binary_output"]
df = pd.DataFrame(all_box_scores, columns=stats)
conn.close()


X = df.drop(columns=["binary_output"]).values
y = df["binary_output"].values
avg_rebounds_index = stats.index("avg_rebounds")
rebound_line_index = stats.index("rebound_line")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class NBAReboundModel(nn.Module):
    def __init__(self, input_dim):
        super(NBAReboundModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        return x


def custom_loss(predictions, targets, avg_rebounds, rebound_line):
    base_loss = nn.BCELoss()(predictions, targets)

    over_penalty = ((avg_rebounds > rebound_line) & (predictions < 0.5)).float() * (avg_rebounds - rebound_line)
    under_penalty = ((avg_rebounds <= rebound_line) & (predictions >= 0.5)).float() * (rebound_line - avg_rebounds)

    penalty = torch.mean(over_penalty + under_penalty)
    return base_loss + penalty

input_size = X_train.shape[1]
model = NBAReboundModel(input_size)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

avg_rebounds_tensor = torch.tensor(X_train[:, avg_rebounds_index], dtype=torch.float32).view(-1, 1)
rebound_line_tensor = torch.tensor(X_train[:, rebound_line_index], dtype=torch.float32).view(-1, 1)

train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)


epochs = 200
best_accuracy = 0

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:
        avg_rebounds_batch = X_batch[:, avg_rebounds_index].view(-1, 1)
        rebound_line_batch = X_batch[:, rebound_line_index].view(-1, 1)

        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = custom_loss(predictions, y_batch, avg_rebounds_batch, rebound_line_batch)
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
        torch.save(best_model_state, 'best_rebound_model.pt')
        print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")


best_model = NBAReboundModel(input_size)
best_model.load_state_dict(torch.load('best_rebound_model.pt'))
best_model.eval()

with torch.no_grad():
    test_predictions = best_model(X_test_tensor)
    test_predictions = test_predictions.round()
    accuracy = (test_predictions.numpy() == y_test_tensor.numpy()).mean()
    print(f"Test Accuracy of Best Model: {accuracy * 100:.2f}%")
