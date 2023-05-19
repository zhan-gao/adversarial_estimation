import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, DataLoader

# Define a Graph Neural Network model
class GNNClassifier(nn.Module):
    def __init__(self, num_features, hidden_size, num_classes):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create a dataset from the adjacency matrices and labels
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __getitem__(self, index):
        graph = self.graphs[index]
        label = self.labels[index]
        edge_index = torch.tensor(graph.nonzero(), dtype=torch.long)
        x = torch.tensor(graph.sum(axis=1), dtype=torch.float)
        return Data(x=x, edge_index=edge_index.t().contiguous(), y=label)

    def __len__(self):
        return len(self.graphs)

# Define hyperparameters and create the model
num_features = n
hidden_size = 16
num_classes = 2
learning_rate = 0.01
num_epochs = 50
batch_size = 1

model = GNNClassifier(num_features, hidden_size, num_classes)

# Create training and testing datasets
train_graphs = [A, B, ..., A]  # List of adjacency matrices for training graphs
train_labels = [0, 1, ..., 0]  # List of labels for training graphs
test_graphs = [A, B, ..., B]  # List of adjacency matrices for testing graphs
test_labels = [0, 1, ..., 1]  # List of labels for testing graphs

train_dataset = GraphDataset(train_graphs, train_labels)
test_dataset = GraphDataset(test_graphs, test_labels)

# Create data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    total_loss = 0
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    print('Epoch {}, Train Loss: {:.4f}'.format(epoch+1, total_loss/len(train_dataset)))

    # Evaluate the model on the test set
    model.eval()
    correct = 0
    for batch in test_loader:
        with torch.no_grad():
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
    acc = correct / len(test_dataset)
    print('Test Accuracy: {:.4f}'.format(acc))