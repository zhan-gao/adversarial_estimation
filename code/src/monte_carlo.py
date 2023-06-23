# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import numpy as np

# %%
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate adjacency matrices for type 0 graphs
num_type0 = 100
type0_matrices = np.random.randint(2, size=(num_type0, 10, 10))

# Generate adjacency matrices for type 1 graphs
num_type1 = 100
type1_matrices = np.random.randint(2, size=(num_type1, 10, 10))

# Concatenate the matrices and labels
adjacency_matrices = np.concatenate([type0_matrices, type1_matrices], axis=0)
labels = np.concatenate([np.zeros(num_type0), np.ones(num_type1)], axis=0)

# Shuffle the data
permutation = np.random.permutation(len(labels))
adjacency_matrices = adjacency_matrices[permutation]
labels = labels[permutation]


# %% 
# Find the maximum number of nodes in the dataset
max_num_nodes = max(matrix.shape[0] for matrix in adjacency_matrices)

# Pad adjacency matrices to have the same number of nodes
padded_matrices = []
for matrix in adjacency_matrices:
    pad_size = max_num_nodes - matrix.shape[0]
    padded_matrix = np.pad(matrix, ((0, pad_size), (0, pad_size)), mode='constant')
    padded_matrices.append(padded_matrix)
padded_matrices = np.array(padded_matrices)

# Convert adjacency matrices to PyTorch Geometric Data objects
data_list = []
for matrix, label in zip(padded_matrices, labels):
    # Remove self-loops (optional)
    matrix = matrix - np.diag(np.diag(matrix))
    
    edge_index = torch.tensor(matrix.nonzero(), dtype=torch.long).t().contiguous()
    x = torch.tensor(matrix, dtype=torch.float).unsqueeze(1)
    y = torch.tensor(label, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data_list.append(data)

# Split the data into training and test sets
num_samples = len(data_list)
num_train = int(0.8 * num_samples)
train_data = data_list[:num_train]
test_data = data_list[num_train:]

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the Graph Convolutional Network (GCN) layer
class GCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x: Node features (adjacency matrix)
        # edge_index: Graph connectivity (indices of neighboring nodes)
        edge_weight = degree(edge_index[1], dtype=torch.float).view(-1, 1)  # Calculate edge weights based on node degrees
        x = self.lin(x)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return x_j * edge_weight

    def update(self, aggr_out):
        return aggr_out

# Define the Graph Classification Model
class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()

        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        x = self.lin(x)
        return x

# Create the model and optimizer
input_dim = 1  # Input dimension is 1 since we have binary adjacency matrices
hidden_dim = 64
num_classes = 2  # Two classes: type 0 and type 1
model = GraphClassifier(input_dim, hidden_dim, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(num_epochs):
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = nn.CrossEntropyLoss()(out, batch.y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    total = 0

    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        _, predicted = torch.max(out.data, 1)
        total += batch.y.size(0)
        correct += (predicted == batch.y).sum().item()

    accuracy = 100 * correct / total
    print('Epoch: {:03d}, Accuracy: {:.2f}%'.format(epoch, accuracy))
