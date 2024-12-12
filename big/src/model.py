import torch.nn as nn
import torch

class TrafficTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, seq_len, output_dim):
        super(TrafficTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
