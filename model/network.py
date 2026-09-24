import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sequence_models.convolutional import ByteNetBlock
from sequence_models.layers import PositionFeedForward


class ByteNetTime(nn.Module):
    """Dilated ByteNet blocks operating on ESM residue representations."""

    def __init__(
        self,
        d_model,
        n_layers,
        kernel_size,
        dilation_cycle,
        causal=False,
        dropout=0.0,
        slim=True,
        activation="relu",
        rank=None,
    ):
        super().__init__()
        cycle_length = int(np.log2(dilation_cycle)) + 1
        hidden_dim = d_model // 2 if slim else d_model
        self.layers = nn.ModuleList(
            ByteNetBlock(
                d_model,
                hidden_dim,
                d_model,
                kernel_size,
                dilation=2 ** (index % cycle_length),
                causal=causal,
                rank=rank,
                activation=activation,
            )
            for index in range(n_layers)
        )
        self.dropout = dropout

    def forward(self, embeddings, input_mask=None):
        for layer in self.layers:
            embeddings = layer(embeddings, input_mask=input_mask)
            if self.dropout > 0.0:
                embeddings = F.dropout(embeddings, p=self.dropout, training=self.training)
        return embeddings


class ByteNetLMTime(nn.Module):
    """Frozen ESM encoder followed by a ByteNet masked-token decoder."""

    def __init__(
        self,
        n_tokens,
        d_model,
        n_layers,
        kernel_size,
        dilation_cycle,
        esm_model,
        causal=False,
        dropout=0.0,
        slim=True,
        activation="relu",
        rank=None,
    ):
        super().__init__()
        self.esm = esm_model
        for parameter in self.esm.parameters():
            parameter.requires_grad = False

        self.embedder = ByteNetTime(
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dilation_cycle=dilation_cycle,
            causal=causal,
            dropout=dropout,
            slim=slim,
            activation=activation,
            rank=rank,
        )
        self.decoder = PositionFeedForward(d_model, n_tokens)

    def forward(self, tokens, input_mask=None):
        embeddings = self.esm(tokens)
        embeddings = self.embedder(embeddings, input_mask=input_mask)
        return self.decoder(embeddings)
