import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class RotaryPositionalEncoding:
    def __init__(self, seq_len: int, head_dim: int, base: float = 10000.0, device: torch.device = None):
        """
        Initialize the rotary positional encoding with the given parameters.

        Args:
            seq_len (int): Maximum sequence length.
            head_dim (int): The dimensionality of each attention head (must be even).
            base (float): The base for the inverse frequency calculation.
            device (torch.device, optional): The device to create the tensors on.
        """
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary positional embeddings.")
        
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device = device
        self.cos, self.sin = self.build_rotary_cache(seq_len, head_dim, base, device)

    def build_rotary_cache(self, seq_len: int, head_dim: int, base: float, device: torch.device):
        """
        Build rotary positional embeddings for a given sequence length and head dimension.

        Returns:
            cos (torch.Tensor): Cosine cache of shape (1, seq_len, 1, head_dim).
            sin (torch.Tensor): Sine cache of shape (1, seq_len, 1, head_dim).
        """
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(seq_len, device=device).float()
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        cos_part = angles.cos()
        sin_part = angles.sin()
        cos = torch.cat([cos_part, cos_part], dim=-1).unsqueeze(0).unsqueeze(2)
        sin = torch.cat([sin_part, sin_part], dim=-1).unsqueeze(0).unsqueeze(2)
        return cos, sin

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper function to rotate the last dimension of the tensor in pairs.

        Args:
            x (torch.Tensor): Tensor of shape (..., head_dim), where head_dim is even.
        
        Returns:
            torch.Tensor: Rotated tensor with the same shape as x.
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embeddings to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, num_heads, head_dim).
        
        Returns:
            torch.Tensor: Tensor with RoPE applied, same shape as x.
        """
        seq_len = x.size(1)
        cos = self.cos[:, :seq_len, ...]
        sin = self.sin[:, :seq_len, ...]
        return x * cos + self.rotate_half(x) * sin

    def example_usage(self):
        """
        Example usage of the RotaryPositionalEncoding class.
        """
        batch_size = 2
        seq_len = 10
        num_heads = 4
        head_dim = 64  # must be even

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=self.device)
        q_rot = self.apply_rotary_pos_emb(q)
        print("q_rot shape:", q_rot.shape)
        return q_rot

# Example usage:
if __name__ == "__main__":
    rope = RotaryPositionalEncoding(seq_len=10, head_dim=64)
    rope.example_usage()

