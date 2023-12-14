import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from graph import create_data

class GATChessModel(torch.nn.Module):
    def __init__(self, num_features, heads=8, dropout=0.6):
        super(GATChessModel, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=heads, dropout=dropout)
        self.conv2 = GATConv(32 * heads, 1, heads=1, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return x

data = create_data()

# Hyperparameters
num_features = 12 * 64
learning_rate = 0.01
num_epochs = 100

model = GATChessModel(num_features)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.MSELoss()  # Mean Squared Error for regression

# Training loop
model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_func(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# if __name__ == '__main__':
#     data = create_data()
#     model = train(data)