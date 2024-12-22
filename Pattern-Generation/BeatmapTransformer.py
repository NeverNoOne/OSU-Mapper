import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from osu_dataset import OsuBeatmapDataset
from torch.cuda.amp import autocast
from torch.amp.grad_scaler import GradScaler
import time
#TODO input additionally the difficulty and bpm 
def collate_fn(batch):
    inputs, targets = zip(*batch)
    
    # Convert inputs to tensors and transpose to [sequence_length, feature_dim]
    inputs = [torch.tensor(x).float().T for x in inputs]  # Transpose: [seq_len, 2]
    targets = [torch.tensor(y).float().unsqueeze(1) for y in targets]  # Targets: [seq_len]
    
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

def compute_loss(predictions, targets, criterion):
    trimmed_predictions = predictions[:, :targets.size(1), :]

    mask = targets != 0

    trimmed_predictions = trimmed_predictions[mask]
    targets = targets[mask]

    loss = criterion(trimmed_predictions, targets)
    return loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    epochs = 10
    batch_size = 16
    model = BeatmapTransformer(input_dim=2, hidden_dim=128, num_layers=4, num_heads=8, output_dim=1).to(device)
    data_loader = DataLoader(OsuBeatmapDataset('Maps'), batch_size=batch_size, collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    bcount = 0
    total_loss = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        begin = time.time()
        scaler = GradScaler()
        for i, batch in enumerate(data_loader):
            bcount += 1
            loss = 0
            x, y = batch  # x: Onset strengths, y: Note placements
            x = x.to(device)
            y = y.to(device)
            mask = create_attention_mask(x)
            with torch.autocast(device.type):                
                predictions = model(x, mask)
                loss = compute_loss(predictions, y, criterion)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            total_loss += loss.item()
            if bcount % 10 == 0: print(f"finished batch: {bcount} with Loss: {loss.item():.4f} took: {time.time()-begin:0.2f} seconds"); begin = time.time()
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(data_loader):.4f}')

    print('saving model...')
    torch.save(model.state_dict(), 'model_weights.pth')
    print('model saved!\nScript is done running')