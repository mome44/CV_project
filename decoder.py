import torch
import torch.nn as nn

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

