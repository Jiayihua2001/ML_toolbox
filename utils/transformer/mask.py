import torch

def PadMask(padded_input, input_lengths=None, pad_idx=None):
    """
    Create a padding mask.
    """
    if padded_input.dim() == 2:
        padded_input = padded_input.unsqueeze(-1)
    if input_lengths is not None:
        N, T, _ = padded_input.shape
        mask = torch.ones((N, T), dtype=torch.bool)
        for i in range(N):
            mask[i, :input_lengths[i]] = False
    else:
        mask = (padded_input.squeeze(-1) == pad_idx)
    return mask.to(padded_input.device)

def CausalMask(input_tensor):
    """
    Create a causal (look‐ahead) mask.
    """
    T = input_tensor.shape[1]
    attn_mask = ~torch.tril(torch.ones((T, T), dtype=torch.bool)).to(input_tensor.device)
    return attn_mask
