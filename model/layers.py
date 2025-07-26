import torch
import torch.nn as nn
import math

# Here be magic

def apply_rotary_pos_emb(x, cos, sin):
    # x: (B, H, T, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    return x_rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for rotary embeddings"
        self.embed_dim = embed_dim

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()  # (batch, time, channels)

        qkv = self.qkv_proj(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # each is (B, T, heads, head_dim)

        q = q.permute(0, 2, 1, 3)  # (B, heads, T, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # q, k: (B, heads, T, head_dim)
        # Prepare rotary embeddings
        theta = 10000 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim).to(x.device)  # (head_dim/2)
        seq_idx = torch.arange(T, device=x.device)  # (T,)
        freqs = torch.einsum("i,j->ij", seq_idx, theta)  # (T, head_dim/2)
        sin, cos = freqs.sin(), freqs.cos()  # (T, head_dim/2)

        # Reshape to (1, 1, T, head_dim/2) for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        # Apply RoPE
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.out_proj(y)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),  # Optionally switch to SwiGLU
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
