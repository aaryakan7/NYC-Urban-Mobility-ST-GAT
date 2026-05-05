import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models import model
from models.model import STGAT
import itertools
from tqdm import tqdm

class NYCTaxiDataset(Dataset):
    def __init__(self, node_features, demand_data, indices, seq_len):
        self.node_features = torch.tensor(node_features, dtype=torch.float32)
        self.demand_data = torch.tensor(demand_data, dtype=torch.float32)
        self.indices = indices
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        # Ensure end_idx doesn't exceed length of demand_data
        if end_idx >= len(self.demand_data):
            end_idx = len(self.demand_data) - 1 
        
        # Removed '+ 1' to close data leak
        # X is seq_len steps 
        X = self.node_features[start_idx : end_idx]  
        
        # Y is target step 
        Y = self.demand_data[end_idx].unsqueeze(-1) 
        
        return X, Y
    
def prepare_data(csv_path, seq_len=48):
    print("Loading processed data...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Separate zone columns from weather columns
    zone_cols = [c for c in df.columns if c.isdigit()]
    weather_cols = [c for c in df.columns if not c.isdigit()]
    
    num_nodes = len(zone_cols)
    num_weather = len(weather_cols)
    num_features = 1 + num_weather
    
    print(f"Detected {num_nodes} zones and {num_weather} weather features.")
    
    # Build array once directly in float32 to save RAM
    time_steps = len(df)
    node_features = np.zeros((time_steps, num_nodes, num_features), dtype=np.float32)
    
    demand_data = df[zone_cols].values.astype(np.float32)
    weather_data = df[weather_cols].values.astype(np.float32)
    
    node_features[:, :, 0] = demand_data
    for w in range(num_weather):
        node_features[:, :, 1 + w] = weather_data[:, w].reshape(-1, 1)
        
    print("Executing chronological split (Jan-May Train / June Test)...")
    valid_indices = np.arange(time_steps - seq_len)
    valid_dates = df.index[valid_indices + seq_len]
    
    # Train on first 5 months of 2024
    train_mask = (valid_dates.year == 2024) & (valid_dates.month <= 5)
    
    # Test on June 2024
    test_mask = (valid_dates.year == 2024) & (valid_dates.month == 6)
    
    train_indices = valid_indices[train_mask]
    test_indices = valid_indices[test_mask]
    
    return node_features, demand_data, train_indices, test_indices, num_nodes, num_features

def train_model():
# Receive flat arrays and indices
    node_features, demand_data, train_indices, test_indices, num_nodes, num_features = prepare_data('processed/final_stgat_input.csv', seq_len=48)
    adj_matrix = np.load('processed/adjacency_matrix.npy')
    
    edges = np.argwhere(adj_matrix == 1)
    edge_index = torch.tensor(edges.T, dtype=torch.long)
    
    # Pass to updated Dataset
    train_dataset = NYCTaxiDataset(node_features, demand_data, train_indices, seq_len=48)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nInitializing model on {device}...")
    edge_index = edge_index.to(device)
    
    # Setup Grid Search
    hidden_dims = [64] 
    learning_rates = [0.001]
    epochs = 30
    
    best_loss = float('inf')
    best_params = {}
    
    # Wrap your original loop in itertools product
    for hidden_dim, lr in itertools.product(hidden_dims, learning_rates):
        print(f"\nTraining Phase 2: hidden_channels={hidden_dim}, lr={lr}, epochs={epochs} ---")
            
        model = STGAT(in_channels=num_features, hidden_channels=hidden_dim, out_channels=1, num_nodes=num_nodes).to(device)
        
        # Swapped to Huber Loss
        criterion = nn.HuberLoss() 
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
        # We can increase step_size since training is 30 epochs 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            
        # Add Learning Rate Scheduler
        scaler = torch.amp.GradScaler('cuda')
            
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # Wrap train_loader with tqdm
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch_X, batch_Y in progress_bar:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                optimizer.zero_grad()
                
                # Autocast for the forward pass 
                with torch.amp.autocast('cuda'):
                    predictions = model(batch_X, edge_index)
                    loss = criterion(predictions, batch_Y)
                
                # Scaled backward pass and optimizer step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
 
                # Update progress bar text with current batch loss
                progress_bar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})
                
            scheduler.step() 
            
            avg_loss = total_loss / len(train_loader)
            current_lr = scheduler.get_last_lr()[0]
            
            # Updated to say 'Huber Loss'
            print(f"Epoch [{epoch+1}/{epochs}] Completed - Avg Huber Loss: {avg_loss:.4f} (LR: {current_lr:.6f})")
            
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = {'hidden_dim': hidden_dim, 'lr': lr}
            torch.save(model.state_dict(), 'models/stgat_weights_best.pth')
            print(f"** New best model saved. Huber Loss: {best_loss:.4f} **")
            
    print(f"\nGrid Search Complete. Best parameters: {best_params} with final Training MSE: {best_loss:.4f}")

if __name__ == "__main__":
    train_model()