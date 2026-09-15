from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionExtractor:
    """
    Hooks into PyTorch TransformerEncoder layers to extract cross-timestep
    attention weight matrices:

        Attention Weights: [B, N, num_heads, T, T]

    Enables physical interpretation of:
        - Which past historical days (lag 1 to 7) dominate forecasting decisions
        - How seasonal variations / storm events alter temporal receptive fields
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights: List[torch.Tensor] = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hooks on MultiheadAttention layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._save_attention_hook)
                self._hooks.append(hook)

    def _save_attention_hook(self, module, input_tensor, output_tensor):
        # MultiheadAttention returns (attn_output, attn_output_weights) if need_weights=True
        if isinstance(output_tensor, tuple) and len(output_tensor) > 1 and output_tensor[1] is not None:
            self.attention_weights.append(output_tensor[1].detach().cpu())

    def clear(self):
        """Clears captured attention matrices."""
        self.attention_weights.clear()

    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def compute_manual_attention_map(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Explicitly computes the query-key dot product attention map
        across the 7-day history window for all ocean nodes.

        Parameters
        ----------
        x : torch.Tensor
            [B, T, N, F]
        edge_index : torch.Tensor
            [2, E]

        Returns
        -------
        attn_matrix : torch.Tensor
            [B, N, T, T] Average attention weight matrix across heads
        """
        self.model.eval()
        B, T, N, in_feat_dim = x.shape

        with torch.no_grad():
            # Project input features
            h = self.model.input_projection(x)
            h_flat = h.reshape(B * T * N, self.model.hidden_dim)
            num_graphs = B * T

            offsets = (
                torch.arange(num_graphs, device=edge_index.device, dtype=torch.long) * N
            ).view(num_graphs, 1, 1)

            batched_edges = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

            h_gnn = torch.relu(self.model.gcn1(h_flat, batched_edges))
            h_gnn = torch.relu(self.model.gcn2(h_gnn, batched_edges))

            h_spatial = h_gnn.reshape(B, T, N, self.model.hidden_dim)
            h_spatial = h_spatial + self.model.time_embedding[:, :T]

            # [B*N, T, H]
            h_node_time = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, self.model.hidden_dim)

            # Query-Key Dot Product Attention:
            # Q * K^T / sqrt(d_k)
            d_k = self.model.hidden_dim
            scores = torch.bmm(h_node_time, h_node_time.transpose(1, 2)) / np.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)  # [B*N, T, T]

            # Reshape back to [B, N, T, T]
            attn_weights = attn_weights.reshape(B, N, T, T)

        return attn_weights

    def get_lag_attribution(self, attn_matrix: torch.Tensor) -> np.ndarray:
        """
        Extracts the attention profile assigned to previous days [Day t-6, ..., Day t]
        when predicting the next state.

        Parameters
        ----------
        attn_matrix : torch.Tensor
            [B, N, T, T]

        Returns
        -------
        lag_weights : np.ndarray
            [T] Normalized average attention given to each history day
        """
        # Focus on the query corresponding to the latest day (index -1)
        # across all batches and nodes
        latest_query_attn = attn_matrix[:, :, -1, :]  # [B, N, T]
        avg_profile = latest_query_attn.mean(dim=(0, 1)).cpu().numpy()
        return avg_profile / avg_profile.sum()
