import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Changed the Learning Rate and epoch higher with less data to try to highr the accuacy
# Using data from an excel sheet this week to use less data whie attempting to fix accuracy 
data = pd.read_excel('./nba_over_under_data.xlsx')

# setting up data sets
X = data[['points_line', 'player_avg_last_season', 'def_team_avg_pts_allowed', 
          'projected_minutes', 'player_avg_at_location', 'opp_def_efficiency',
          'usage_rate', 'historical_perf_vs_opponent']].values
y = data['hit_over'].values.reshape(-1, 1)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# initialize weights and biases used chatgpt for the random feature
input_size = X_train_scaled.shape[1]
hidden_layer1_size = 64
hidden_layer2_size = 32
output_size = 1

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_layer1_size) * 0.01 
b1 = np.zeros((1, hidden_layer1_size))  

W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01 
b2 = np.zeros((1, hidden_layer2_size))  

W3 = np.random.randn(hidden_layer2_size, output_size) * 0.01  
b3 = np.zeros((1, output_size))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# forward  propagation using all three layers
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)  
    
    return Z1, A1, Z2, A2, Z3, A3

# used help from chatgpt to construct this with an explaination
def compute_cost(A3, y):
    m = y.shape[0]
    cost = -1/m * np.sum(y * np.log(A3) + (1 - y) * np.log(1 - A3))
    return cost

# 9. back  propagation
def backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = X.shape[0]
    
    # Output layer 
    dZ3 = A3 - y
    dW3 = 1/m * np.dot(A2.T, dZ3)
    db3 = 1/m * np.sum(dZ3, axis=0, keepdims=True)
    
    # Hidden layer 
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = 1/m * np.dot(A1.T, dZ2)
    db2 = 1/m * np.sum(dZ2, axis=0, keepdims=True)
    
    # Hidden layer 1 gradients
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = 1/m * np.dot(X.T, dZ1)
    db1 = 1/m * np.sum(dZ1, axis=0, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3


def update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate=0.5):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    
    return W1, b1, W2, b2, W3, b3

#  training model 
def train_model(X, y, W1, b1, W2, b2, W3, b3, epoch=1000, learning_rate=0.5):
    for i in range(epoch):

        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X, y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3)
        W1, b1, W2, b2, W3, b3 = update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
    
    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = train_model(X_train_scaled, y_train, W1, b1, W2, b2, W3, b3, epoch=900, learning_rate=0.5)


def predict(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
    predictions = (A3 > 0.4).astype(int)
    return predictions

predictions = predict(X_test_scaled, W1, b1, W2, b2, W3, b3)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy}")
