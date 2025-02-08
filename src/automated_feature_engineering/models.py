"""
Autoencoder with skip connections for automatic feature extraction.
"""
import torch
from torch import nn


class Autoencoder(nn.Module):
    """Autoencoder with skip connections for automatic feature extraction."""

    def __init__(self, input_dim: int):
        super().__init__()

        # Encoder
        self.enc1 = nn.Linear(input_dim, 128)
        self.enc_act1 = nn.ReLU()
        self.enc2 = nn.Linear(128, 64)
        self.enc_act2 = nn.ReLU()
        self.enc3 = nn.Linear(64, 32)
        self.enc_act3 = nn.ReLU()
        self.enc4 = nn.Linear(32, 16)

        self.layer_norm1 = nn.LayerNorm(128)
        self.layer_norm2 = nn.LayerNorm(64)
        self.layer_norm3 = nn.LayerNorm(32)


        # Decoder with skip connections
        self.dropout = nn.Dropout(0.2)
        self.dec1 = nn.Linear(16, 32)
        self.dec_act1 = nn.ReLU()
        self.dec2 = nn.Linear(32, 64)
        self.dec_act2 = nn.ReLU()
        self.dec3 = nn.Linear(64, 128)
        self.dec_act3 = nn.ReLU()
        self.dec4 = nn.Linear(128, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        # Encoder
        enc1_out = self.enc_act1(self.enc1(x))
        enc2_out = self.enc_act2(self.enc2(enc1_out))
        enc3_out = self.enc_act3(self.enc3(enc2_out))
        encoding = self.enc4(enc3_out)
        drop = self.dropout(encoding)

        # Decoder with skip connections
        dec1_out = self.dec_act1(self.dec1(drop))
        dec2_in = self.layer_norm3(dec1_out + enc3_out) # Skip connection
        dec2_out = self.dec_act2(self.dec2(dec2_in))
        dec3_in = self.layer_norm2(dec2_out + enc2_out)
        dec3_out = self.dec_act3(self.dec3(dec3_in))
        dec4_in = self.layer_norm1(dec3_out + enc1_out)
        output = self.dec4(dec4_in)

        return output


class Encoder(nn.Module):
    """Encoder component extraction"""

    def __init__(self, autoencoder: Autoencoder):
        super().__init__()
        self.enc1 = autoencoder.enc1
        self.enc_act1 = autoencoder.enc_act1
        self.enc2 = autoencoder.enc2
        self.enc_act2 = autoencoder.enc_act2
        self.enc3 = autoencoder.enc3
        self.enc_act3 = autoencoder.enc_act3
        self.enc4 = autoencoder.enc4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        x = self.enc_act1(self.enc1(x))
        x = self.enc_act2(self.enc2(x))
        x = self.enc_act3(self.enc3(x))
        return self.enc4(x)

