import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class STGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_nodes):
        super(STGAT, self).__init__()
        # Spatial Layer for GAT
        # Learns importance of neighboring zones based on real-time features
        self.gat = GATv2Conv(in_channels, hidden_channels, heads=4, concat=False)

        # Temporal Layer for GRU
        # Learns time-series trends over last few hours
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        # Fully Connected layer to predict demand for next time step
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, num_features = x.size()

        # Flatten time and batch dimensions to push through graph layer
        x_reshaped = x.view(-1, num_features)

        # Apply GAT and a ReLU activation
        out_gat = self.gat(x_reshaped, edge_index)
        out_gat = F.relu(out_gat)

        # Reshape back to organize by time sequence for the GRU
        out_gat = out_gat.view(batch_size, seq_len, num_nodes, -1)

        # Change dimensions for GRU
        out_gat = out_gat.permute(0, 2, 1, 3).contiguous()
        out_gat = out_gat.view(batch_size * num_nodes, seq_len, -1)

        # Pass through GRU
        out_gru, _ = self.gru(out_gat)

        # Grab output of last time step to make the future prediction
        last_step = out_gru[:, -1, :]
            
        # Final output layer
        out = self.linear(last_step)

        # Reshape
        return out.view(batch_size, num_nodes, -1)
