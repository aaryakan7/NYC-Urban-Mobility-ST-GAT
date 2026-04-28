import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.model import STGAT

class NYCTaxiDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
def prepare_data(csv_path, seq_len=12):
    print("Loading processed data...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Separate the zone columns from the weather columns
    zone_cols = [c for c in df.columns if c.isdigit()]
    weather_cols = [c for c in df.columns if not c.isdigit()]
    
    num_nodes = len(zone_cols)
    num_weather = len(weather_cols)
    num_features = 1 + num_weather
    
    print(f"Detected {num_nodes} zones and {num_weather} weather features.")
    
    # Reshape the flat table into a 3D tensor
    time_steps = len(df)
    node_features = np.zeros((time_steps, num_nodes, num_features))
    
    demand_data = df[zone_cols].values
    weather_data = df[weather_cols].values
    
    node_features[:, :, 0] = demand_data
    for w in range(num_weather):
        # Broadcast the weather feature across all nodes for each time step
        node_features[:, :, 1 + w] = weather_data[:, w].reshape(-1, 1)
        
    # Create the sliding windows
    print("Creating sequential windows...")
    X, Y, valid_dates = [], [], []
    for i in range(time_steps - seq_len):
        X.append(node_features[i : i + seq_len])
        # The target is just the demand of the next time step
        Y.append(demand_data[i + seq_len]) 
        valid_dates.append(df.index[i + seq_len])
        
    X = np.array(X)
    Y = np.expand_dims(np.array(Y), axis=-1)
    dates = pd.DatetimeIndex(valid_dates)
    
    # Train on 2024, Test on 2025
    print("Executing chronological split (2024 Train / 2025 Test)...")
    train_mask = dates.year == 2024
    test_mask = dates.year == 2025
    
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    
    return X_train, Y_train, X_test, Y_test, num_nodes, num_features

def train_model():
    X_train, Y_train, X_test, Y_test, num_nodes, num_features = prepare_data('processed/final_stgat_input.csv', seq_len=12)
    adj_matrix = np.load('processed/adjacency_matrix.npy')
    
    # Convert adjacency matrix to edge index format for PyTorch Geometric
    edges = np.argwhere(adj_matrix == 1)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    train_dataset = NYCTaxiDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing model on {device}...")
    
    model = STGAT(in_channels=num_features, hidden_channels=64, out_channels=1, num_nodes=num_nodes).to(device)
    edge_index = edge_index.to(device)
    
    # Optimization setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 10
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X, edge_index)
            
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Training MSE Loss: {avg_loss:.4f}")
        
    # Save the trained model weights
    torch.save(model.state_dict(), 'models/stgat_weights.pth')
    print("Training complete! Model saved to models/stgat_weights.pth.")

if __name__ == "__main__":
    train_model()
