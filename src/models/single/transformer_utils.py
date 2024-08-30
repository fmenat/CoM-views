import torch
from torch import nn, Tensor

class PositionalEncoding(nn.Module):
    """
    ref: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding

    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # create a long enough position encoding
        position_encoding = torch.zeros(1, max_len, d_model, )

        position = torch.arange(
            max_len, dtype=torch.float32
        ).reshape(-1, 1) / torch.pow(torch.tensor(10000), torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)

        position_encoding[0, :, 0::2] = torch.sin(position)
        position_encoding[0, :, 1::2] = torch.cos(position)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("position_encoding", position_encoding, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.position_encoding[:, :x.shape[1], :]  # based on the seq len of x, add position encoding
        return self.dropout(x)