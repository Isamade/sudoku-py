import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SudokuGNN(nn.Module):
    def __init__(self, num_features=5, hidden_dim=64, num_classes=5):
        """
        GNN for solving 4x4 Sudoku.
        
        Args:
            num_features: Number of input features per node (1 current value + 4 row/col/box indicators = 5)
                          Actually, a simpler approach: node value (0-4) as input.
                          Let's use an embedding for the input digit.
            hidden_dim: Dimension of hidden layers.
            num_classes: Number of output classes (0-4, though usually 1-4).
        """
        super(SudokuGNN, self).__init__()
        
        # Embedding for input values (0 for empty, 1-4 for given)
        self.embedding = nn.Embedding(5, hidden_dim)
        
        # Graph Attention Layers
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
        
        # Fully connected layer for the final classification
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # x shape: (num_nodes, 1) -> containing values 0-4
        x = x.squeeze().long()
        x = self.embedding(x) # (num_nodes, hidden_dim)
        
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Third GAT layer
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        
        # Final prediction
        logits = self.fc(x) # (num_nodes, 5) -> predicts probability of digits 0-4
        
        return logits
