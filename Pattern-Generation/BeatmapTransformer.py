import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from osu_dataset import OsuBeatmapDataset

def collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # Convert inputs to tensors and transpose to [sequence_length, feature_dim]
    inputs = [torch.tensor(x).float().T for x in inputs]  # Transpose: [seq_len, 2]
    targets = [torch.tensor(y).float() for y in targets]  # Targets: [seq_len]
    
    # Pad sequences to match the longest sequence in the batch
    inputs_padded = pad_sequence(inputs, batch_first=True)  # [batch_size, max_seq_len, 2]
    targets_padded = pad_sequence(targets, batch_first=True)  # [batch_size, max_seq_len]
    
    return inputs_padded, targets_padded

class BeatmapTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim):
        super(BeatmapTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)  # Project input to hidden_dim
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask=None):
        x = self.input_projection(x)  # Shape: [batch_size, seq_len, hidden_dim]
        x = x.permute(1, 0, 2)  # Shape: [seq_len, batch_size, hidden_dim] (required by Transformer)
        encoded = self.encoder(x, src_key_padding_mask=mask)
        encoded = encoded.permute(1, 0, 2)  # Shape: [batch_size, seq_len, hidden_dim]
        output = self.decoder(encoded)  # Shape: [batch_size, seq_len, output_dim]
        return output

def create_attention_mask(padded_inputs):
    # Mask padded values (assume padded values are 0)
    return (padded_inputs.sum(dim=-1) == 0)  # Shape: [batch_size, max_seq_len]

if __name__ == '__main__':
    epochs = 2
    model = BeatmapTransformer(input_dim=2, hidden_dim=128, num_layers=4, num_heads=8, output_dim=1)
    data_loader = DataLoader(OsuBeatmapDataset('Maps', count_mapsets=2), batch_size=32, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        for batch in data_loader:
            x, y = batch  # x: Onset strengths, y: Note placements
            optimizer.zero_grad()
            mask = create_attention_mask(x)
            predictions = model(x, mask)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

# Exception has occurred: TypeError
# default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'osu.HitObject'>
# TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'osu.HitObject'>

# During handling of the above exception, another exception occurred:

#   File "C:\Users\maxhe\source\repos\github\OSU-Mapper\Pattern-Generation\BeatmapTransformer.py", line 29, in <module>
#     for batch in data_loader:
#                  ^^^^^^^^^^^
# TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'osu.HitObject'>