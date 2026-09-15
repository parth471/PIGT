from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GaussianNLLLoss(nn.Module):
    """
    Heteroscedastic Gaussian Negative Log-Likelihood (NLL) Loss:
        Loss(μ, s, y) = 0.5 * ( exp(-s) * (y - μ)² + s )
    Where:
        - μ is the predictive mean
        - s = log(σ²) is the predicted log-variance (aleatoric uncertainty)
        - y is the ground truth target
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        mean: torch.Tensor,
        log_var: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        precision = torch.exp(-log_var)

        loss = 0.5 * (precision * (target - mean) ** 2 + log_var)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class PIGTUncertaintyModel(nn.Module):
    """
    Physics-Informed Graph Transformer with Calibrated Uncertainty Quantification (UQ).

    Dual Prediction Head:
        1. Mean Head: μ (Predicted SST) [B, N]
        2. Aleatoric Variance Head: s = log(σ²) [B, N]

    Supports Monte Carlo (MC) Dropout at inference time to decompose:
        Total Uncertainty = Aleatoric (Data noise) + Epistemic (Model uncertainty)
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
        self.dropout_rate = dropout
        self.max_time_steps = max_time_steps

        # 1. Feature Projection
        self.input_projection = nn.Linear(in_features, hidden_dim)

        # 2. Spatial Graph Convolutional Layers
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # 3. Learnable Temporal Positional Embedding
        self.time_embedding = nn.Parameter(
            torch.zeros(1, max_time_steps, 1, hidden_dim)
        )

        # 4. Temporal Transformer Encoder
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

        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # 5. Shared Representation Layer
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 6a. Mean SST Prediction Head: μ [B, N, 1]
        self.mean_head = nn.Linear(hidden_dim, 1)

        # 6b. Log-Variance Aleatoric Uncertainty Head: s = log(σ²) [B, N, 1]
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            [B, T, N, F] (Batch, History Window 7, Ocean Nodes 12204, Features 13)
        edge_index : torch.Tensor
            [2, E] Graph edge connectivity

        Returns
        -------
        mean : torch.Tensor
            [B, N] Predicted next-day SST (μ)
        log_var : torch.Tensor
            [B, N] Predicted log-variance (log σ²)
        """
        B, T, N, in_feat_dim = x.shape

        # Step 1: Project input features [B, T, N, F] -> [B, T, N, H]
        h = self.input_projection(x)

        # Step 2: Batched Spatial GNN over all B*T graph snapshots
        h_flat = h.reshape(B * T * N, self.hidden_dim)
        num_graphs = B * T

        offsets = (
            torch.arange(num_graphs, device=edge_index.device, dtype=torch.long) * N
        ).view(num_graphs, 1, 1)

        batched_edges = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

        h_gnn = torch.relu(self.gcn1(h_flat, batched_edges))
        h_gnn = torch.relu(self.gcn2(h_gnn, batched_edges))

        h_spatial = h_gnn.reshape(B, T, N, self.hidden_dim)

        # Step 3: Add temporal positional embeddings
        h_spatial = h_spatial + self.time_embedding[:, :T]

        # Step 4: Temporal Transformer per node [B*N, T, H]
        h_node_time = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        h_trans = self.temporal_transformer(h_node_time)

        # Extract latest timestep representation
        h_latest = self.temporal_norm(h_trans[:, -1, :]).reshape(B, N, self.hidden_dim)

        # Step 5: Shared Head
        h_shared = self.shared_head(h_latest)

        # Step 6: Dual Head Outputs
        mean = self.mean_head(h_shared).squeeze(-1)       # [B, N]
        log_var = self.log_var_head(h_shared).squeeze(-1) # [B, N]

        return mean, log_var

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_mc_samples: int = 20,
    ) -> Dict[str, torch.Tensor]:
        """
        Performs Monte Carlo (MC) Dropout inference to decompose aleatoric
        and epistemic uncertainty.

        Returns
        -------
        dict with:
            - 'mean': [B, N] Expected SST prediction
            - 'aleatoric_var': [B, N] Data/sensor noise variance
            - 'epistemic_var': [B, N] Model parameter variance
            - 'total_var': [B, N] Combined uncertainty
            - 'std': [B, N] Standard deviation (confidence bounds)
        """
        # Enable dropout during inference for MC sampling
        self.train()

        means = []
        log_vars = []

        with torch.no_grad():
            for _ in range(num_mc_samples):
                mu, s = self.forward(x, edge_index)
                means.append(mu)
                log_vars.append(s)

        # Stack samples: [M, B, N]
        means_stack = torch.stack(means, dim=0)
        log_vars_stack = torch.stack(log_vars, dim=0)

        # 1. Predictive Mean: E[μ]
        mean_pred = torch.mean(means_stack, dim=0)  # [B, N]

        # 2. Aleatoric Uncertainty: E[σ²] = E[exp(s)]
        aleatoric_var = torch.mean(torch.exp(log_vars_stack), dim=0)  # [B, N]

        # 3. Epistemic Uncertainty: Var(μ) = E[μ²] - (E[μ])²
        epistemic_var = torch.var(means_stack, dim=0, unbiased=True)  # [B, N]

        # 4. Total Uncertainty
        total_var = aleatoric_var + epistemic_var
        total_std = torch.sqrt(total_var)

        self.eval()

        return {
            "mean": mean_pred,
            "aleatoric_var": aleatoric_var,
            "epistemic_var": epistemic_var,
            "total_var": total_var,
            "std": total_std,
        }
