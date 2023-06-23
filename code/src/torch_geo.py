# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# %% 
# Define the Graph Convolutional Network (GCN) model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Load the Cora dataset from torch_geometric
dataset = Planetoid(root='/tmp/Cora', name='Cora')

# Create data loaders for training and testing
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset, batch_size=128, shuffle=False)

# Initialize the model
model = GCN(dataset.num_features, 16, dataset.num_classes)

# Set the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
model.train()
for epoch in range(200):
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    # Calculate training accuracy
    _, predicted = model(data.x, data.edge_index).max(dim=1)
    correct = predicted[data.test_mask] == data.y[data.test_mask]
    acc = int(correct.sum()) / int(data.test_mask.sum())

    print(f'Epoch: {epoch+1:03d}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}')

# Evaluation on the test set
model.eval()
for data in test_loader:
    out = model(data.x, data.edge_index)
    _, predicted = out.max(dim=1)
    correct = predicted[data.test_mask] == data.y[data.test_mask]
    test_acc = int(correct.sum()) / int(data.test_mask.sum())

print(f'Test Accuracy: {test_acc:.4f}')