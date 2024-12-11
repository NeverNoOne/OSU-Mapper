import torch
import torch.nn as nn

class BeatmapTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim):
        super(BeatmapTransformer, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output