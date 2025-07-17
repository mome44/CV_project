import torch
import torch.nn as nn

# Positional Encoding (apprendibile, semplice)
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=108):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim))  # [1, 108, 512]

    def forward(self, x):
        return x + self.pos_embed  # broadcasting over batch

# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, dim=512, inner_dim=1024, n_heads=8):
        super().__init__()
        self.expand = nn.Conv1d(dim, inner_dim, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=inner_dim, num_heads=n_heads, batch_first=True)
        self.reduce = nn.Conv1d(inner_dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, 108, 512]
        x_in = x
        x = x.transpose(1, 2)  # [B, 512, 108]
        x = self.expand(x)     # [B, 1024, 108]
        x = x.transpose(1, 2)  # [B, 108, 1024]

        attn_out, _ = self.attn(x, x, x)  # self-attention
        x = self.reduce(attn_out.transpose(1, 2)).transpose(1, 2)  # back to [B, 108, 512]
        x = self.norm(x + x_in)  # residual + norm
        return x

# Encoder (3 blocchi)
class PDLPR_Encoder(nn.Module):
    def __init__(self, dim=512, n_heads=8, depth=3):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim)
        self.blocks = nn.Sequential(*[EncoderBlock(dim, 1024, n_heads) for _ in range(depth)])

    def forward(self, x):
        # x: [B, 512, 6, 18]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)              # [B, 512, 108] -> shape [B, C, N] (Convolutional layers expects this)
        # reorder dimentions for Transformer
        x = x.permute(0, 2, 1)               # [B, 108, 512] -> changes to shape [B, N, C] (Transformer expects this)
        x = self.pos_enc(x)                  # Add positional encoding
        x = self.blocks(x)                   # Encoder blocks
        return x  # [B, 108, 512]
