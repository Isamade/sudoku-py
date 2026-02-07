import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import SudokuGNN
from data_loader import SudokuDataset
import os

def train():
    # Parameters
    num_samples = 5000
    batch_size = 32
    hidden_dim = 128
    epochs = 20
    learning_rate = 0.001
    model_path = "sudoku_gnn.pth"
    
    # Initialize Dataset and DataLoader
    print(f"Generating {num_samples} Sudoku puzzles...")
    dataset = SudokuDataset(num_samples=num_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize Model, Optimizer, and Loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SudokuGNN(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_nodes = 0
        
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            
            # Loss relative to the ground truth solution
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total_nodes += data.y.size(0)
            
        avg_loss = total_loss / len(loader)
        accuracy = correct / total_nodes
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
