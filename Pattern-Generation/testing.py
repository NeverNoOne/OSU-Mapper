import torch
import torch.nn as nn
from BeatmapTransformer import BeatmapTransformer

model = BeatmapTransformer(input_dim=1, hidden_dim=128, num_layers=4, num_heads=8, output_dim=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 1

for epoch in range(epochs):
    model.train()
    for batch in data_loader:
        x, y = batch #x: onset strengths, y: note placements
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions,y)
        loss.backward()
        optimizer.step()