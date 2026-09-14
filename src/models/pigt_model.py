import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class PIGTModel(nn.Module):
    """
    Graph + Temporal Transformer model for next-day SST forecasting.

    Input:
        X: [B, T, N, F]

        B = batch size
        T = history length (7)
        N = number of ocean nodes (12204)
        F = number of input features (13)

    Output:
        Future SST: [B, N]
    """

    def __init__(
        self,
        in_features: int = 13,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        max_time_steps: int = 7,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim

        # ----------------------------------------------------
        # Input feature projection
        # ----------------------------------------------------

        self.input_projection = nn.Linear(
            in_features,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Spatial GNN
        # ----------------------------------------------------

        self.gcn1 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        self.gcn2 = GCNConv(
            hidden_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Temporal Transformer
        # ----------------------------------------------------

        self.time_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_time_steps,
                1,
                hidden_dim,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        self.temporal_norm = nn.LayerNorm(
            hidden_dim
        )

        # ----------------------------------------------------
        # SST prediction head
        # ----------------------------------------------------

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            [B, T, N, F]

        edge_index:
            [2, E]

        Returns
        -------
        prediction:
            [B, N]
        """

        B, T, N, F = x.shape

        if F != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, "
                f"got {F}"
            )

        # ----------------------------------------------------
        # 1. Project input features
        #
        # [B,T,N,F]
        #       ↓
        # [B,T,N,H]
        # ----------------------------------------------------

        x = self.input_projection(x)

                # ----------------------------------------------------
        # 2. Apply GNN independently to every timestep
        #    using one batched graph operation.
        #
        # [B,T,N,H]
        #       ↓
        # [B*T*N,H]
        #
        # We create B*T copies of the same graph with
        # different node-index offsets.
        # ----------------------------------------------------

        x = x.reshape(
            B * T * N,
            self.hidden_dim,
        )

        # Number of graph copies
        num_graphs = B * T

        # Original graph:
        # [2, E]
        #
        # Create node-index offsets:
        # [0, N, 2N, ..., (B*T-1)N]
        offsets = (
            torch.arange(
                num_graphs,
                device=edge_index.device,
                dtype=torch.long,
            )
            * N
        )

        # [B*T, 1, 1]
        offsets = offsets.view(
            num_graphs,
            1,
            1,
        )

        # [B*T, 2, E]
        batched_edge_index = (
            edge_index.unsqueeze(0)
            + offsets
        )

        # [2, B*T*E]
        batched_edge_index = (
            batched_edge_index
            .permute(1, 0, 2)
            .reshape(
                2,
                -1,
            )
        )

        # ----------------------------------------------------
        # First GCN layer
        # ----------------------------------------------------

        x = self.gcn1(
            x,
            batched_edge_index,
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # Second GCN layer
        # ----------------------------------------------------

        x = self.gcn2(
            x,
            batched_edge_index,
        )

        x = torch.relu(x)

        # ----------------------------------------------------
        # Restore:
        #
        # [B*T*N,H]
        #       ↓
        # [B,T,N,H]
        # ----------------------------------------------------

        x = x.reshape(
            B,
            T,
            N,
            self.hidden_dim,
        )

        # ----------------------------------------------------
        # Restore time dimension
        #
        # [B*T,N,H]
        #       ↓
        # [B,T,N,H]
        # ----------------------------------------------------

        x = x.reshape(
            B,
            T,
            N,
            self.hidden_dim,
        )

        # ----------------------------------------------------
        # 3. Add temporal positional information
        # ----------------------------------------------------

        if T > self.time_embedding.shape[1]:
            raise ValueError(
                f"Received {T} timesteps but model supports "
                f"maximum {self.time_embedding.shape[1]}"
            )

        x = (
            x
            + self.time_embedding[:, :T]
        )

        # ----------------------------------------------------
        # 4. Temporal Transformer
        #
        # Each node gets its own temporal sequence.
        #
        # [B,T,N,H]
        #       ↓
        # [B,N,T,H]
        #       ↓
        # [B*N,T,H]
        # ----------------------------------------------------

        x = x.permute(
            0,
            2,
            1,
            3,
        )

        x = x.reshape(
            B * N,
            T,
            self.hidden_dim,
        )

        x = self.temporal_transformer(
            x
        )

        # ----------------------------------------------------
        # Use representation of latest time step
        # ----------------------------------------------------

        x = x[:, -1, :]

        x = self.temporal_norm(
            x
        )

        # ----------------------------------------------------
        # Restore node dimension
        # ----------------------------------------------------

        x = x.reshape(
            B,
            N,
            self.hidden_dim,
        )

        # ----------------------------------------------------
        # 5. Predict next-day SST
        #
        # [B,N,H]
        #       ↓
        # [B,N,1]
        #       ↓
        # [B,N]
        # ----------------------------------------------------

        prediction = self.prediction_head(
            x
        )

        prediction = prediction.squeeze(-1)

        return prediction