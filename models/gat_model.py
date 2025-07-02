import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATLayoutPredictor(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, heads=4, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.node_embedding = nn.Embedding(num_node_features, hidden_dim)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            self.convs.append(GATConv(in_channels, hidden_dim, heads=heads, edge_dim=num_edge_features))

        # Final fully connected layer to predict layout parameters
        self.fc = nn.Linear(hidden_dim * heads, 4)  # Predict x, y, width, height
        
    def forward(self, data):
        device = next(self.parameters()).device
        
        # Handle node features
        x = data.x.long().to(device)
        x = self.node_embedding(x).squeeze(1)  # Remove extra dimension if needed
        
        # Handle edge features
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr.float().to(device)
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)  # Ensure 2D shape
                
        edge_index = data.edge_index.to(device)
        
        # Apply GAT layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
        
        # Global pooling and prediction
        return torch.sigmoid(self.fc(x))