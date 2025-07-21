import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

#DEFINITION MODEL CLASSES FOR PDLPR

#igfe feature extractor
class FocusStructure(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(FocusStructure, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        patch_tl = x[..., ::2, ::2]  
        patch_tr = x[..., ::2, 1::2]  
        patch_bl = x[..., 1::2, ::2]  
        patch_br = x[..., 1::2, 1::2] 
        x = torch.cat([patch_tl, patch_tr, patch_bl, patch_br], dim=1)  # [B, 4C, H/2, W/2]
        return self.conv(x)


class ConvDownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDownSampling, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1) #here we use as activation function LeakyReLU, which is more used in car plate detection
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.relu(x + self.block(x))


class IGFE(nn.Module):
    def __init__(self):
        super(IGFE, self).__init__()
        self.focus = FocusStructure(3, 64)         # From [3,48,144] to [64,24,72]
        self.down1 = ConvDownSampling(64, 128)     # [128,12,36]
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.down2 = ConvDownSampling(128, 256)    # [256,6,18]
        self.res3 = ResBlock(256)
        self.res4 = ResBlock(256)
        self.final_conv = nn.Conv2d(256, 512, kernel_size=1)  # [512,6,18]

    def forward(self, x):
        x = self.focus(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.final_conv(x)
        return x  # [B, 512, 6, 18]
    
#   encoder
# Positional Encoding 
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

# Encoder (3 blocks)
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


  #parallel decoder


#  decoder

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ff = FeedForward(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: [B, T, dim], memory: [B, S, dim]
        x = tgt

        # masked self attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + attn_out)

        # cross attention
        attn_out, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + attn_out)

        # feedforward neural network
        ff_out = self.ff(x)
        x = self.norm3(x + ff_out)

        return x

class ParallelDecoder(nn.Module):
    def __init__(self, dim=512, vocab_size=70, num_heads=8, num_blocks=3, seq_len=18):
        super().__init__()
        self.seq_len = seq_len
        self.char_embed = nn.Parameter(torch.randn(1, seq_len, dim)) 
        self.vocab_size = vocab_size
        self.dim = dim

        self.blocks = nn.ModuleList([
            DecoderBlock(dim, num_heads) for _ in range(num_blocks)
        ])
        self.classifier = nn.Linear(dim, vocab_size)
    
    def update_vocab_size(self, new_vocab_size):
         
         if new_vocab_size != self.vocab_size:
            print(f"Updating vocab size from {self.vocab_size} to {new_vocab_size}")
            # Save old weights 
            old_classifier = self.classifier
            old_out_features = old_classifier.out_features
        
            # Create new classifier
            new_classifier = nn.Linear(self.dim, new_vocab_size)
            new_classifier = new_classifier.to(old_classifier.weight.device)
        
            # Copy overlapping weights
            num_to_copy = min(old_out_features, new_vocab_size)
            with torch.no_grad():
                new_classifier.weight[:num_to_copy] = old_classifier.weight[:num_to_copy]
                new_classifier.bias[:num_to_copy] = old_classifier.bias[:num_to_copy]
        
            self.classifier = new_classifier
            self.vocab_size = new_vocab_size

    def generate_mask(self, size):
        # mask future tokens
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, memory):
        # memory: [B, S, dim] → encoder output (B, 108, 512])
        B = memory.size(0)
        x = self.char_embed.expand(B, -1, -1)  # [B, T, dim]
        tgt_mask = self.generate_mask(self.seq_len).to(memory.device)  # [T, T]

        for block in self.blocks:
            x = block(x, memory, tgt_mask)

        logits = self.classifier(x)  # [B, T, vocab_size]
        return logits


#DEFINITION MODEL CLASS FOR BASELINE METHOD

class CNN_CTC_model(nn.Module):
    def __init__(self, num_char, hidden_size):
        super(CNN_CTC_model, self).__init__()
        self.num_char = num_char
        self.hidden_size = hidden_size
        #self.final_feature_width = final_feature_width
                
        self.features = nn.Sequential(
            #1 Because we use grayscale images
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2, 2),                                     

            nn.Conv2d(128, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )  

        #two linear layers to do the final classification
        self.linear = nn.Linear(self.hidden_size * 12, 256)  # 256×12 = concatenazione H dim
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.hidden_size, self.num_char)

    def forward(self, x):
        #the input size is:  Batch, 1, 48 x 144
        x = self.features(x)   #output size: Batch, 256, 12x36

        #since we have 4 elements, the CTC wants the width first so we have to 
        #put it into the first position
        x = x.permute(3, 0, 1, 2)  # 36, batch, 256, 12 
        #the width so the frames must be more than the number of total characters that
        #we want to encode, so T = width = 36
        x = x.flatten(2)          # 36, batch , 256×12]
        x = self.linear(x)
        x = self.dropout(x)           # 36, batch, 256
        x = self.classifier(x)    # 36, batch, num_char

        return x  #returns a tensor of size [numchar] for each one of the 36 positions
