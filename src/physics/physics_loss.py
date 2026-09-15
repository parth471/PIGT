from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OceanAdvectionDiffusionLoss(nn.Module):
    """
    Physics-Informed Loss implementing the 2D Ocean Thermal Advection-Diffusion
    conservation equation on irregular spatial graph structures:

        ∂T/∂t + (u · ∇)T = κ ∇²T

    Where:
        - T is Sea Surface Temperature (SST)
        - u = (u_current, v_current) is the horizontal velocity vector
        - κ is the horizontal thermal diffusivity coefficient
        - (u · ∇)T is thermal advection driven by ocean currents
        - κ ∇²T is lateral thermal diffusion across neighboring nodes

    Tensor shapes:
        - pred_sst_next: [B, N] (predicted SST at t+1)
        - x_current_state: [B, N, F] or full history [B, T, N, F]
        - edge_index: [2, E]
        - node_coords: [N, 2] (lat, lon coordinates in degrees or radians)
    """

    def __init__(
        self,
        diffusivity_kappa: float = 1e-3,
        delta_t_days: float = 1.0,
        sst_idx: int = 0,
        u_current_idx: int = 2,
        v_current_idx: int = 3,
        lat_idx: int = 9,
        lon_idx: int = 10,
        reduction: str = "mean",
    ):
        super().__init__()
        self.diffusivity_kappa = diffusivity_kappa
        self.delta_t = delta_t_days
        self.sst_idx = sst_idx
        self.u_idx = u_current_idx
        self.v_idx = v_current_idx
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.reduction = reduction

    def compute_graph_spatial_derivatives(
        self,
        temperature: torch.Tensor,
        u_velocity: torch.Tensor,
        v_velocity: torch.Tensor,
        edge_index: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the advection term (u · ∇T) and diffusion term (κ ∇²T)
        for every node using discrete spatial graph operators.

        Parameters
        ----------
        temperature : torch.Tensor
            [B, N] SST field at current time step
        u_velocity : torch.Tensor
            [B, N] Zonal eastward velocity
        v_velocity : torch.Tensor
            [B, N] Meridional northward velocity
        edge_index : torch.Tensor
            [2, E] Directed graph connectivity
        coords : torch.Tensor, optional
            [N, 2] Spatial coordinates (lat, lon)

        Returns
        -------
        advection : torch.Tensor [B, N]
        diffusion : torch.Tensor [B, N]
        """
        B, N = temperature.shape
        src, dst = edge_index[0], edge_index[1]  # src -> dst
        E = edge_index.shape[1]

        # Temperature differences along directed edges: T_j - T_i
        # [B, E]
        temp_src = temperature[:, src]
        temp_dst = temperature[:, dst]
        delta_temp = temp_dst - temp_src

        # Coordinate differences if available, otherwise uniform topology
        if coords is not None:
            # coords: [N, 2] -> (lat, lon)
            coords_src = coords[src]  # [E, 2]
            coords_dst = coords[dst]  # [E, 2]
            delta_coords = coords_dst - coords_src  # [E, 2]

            d_lat = delta_coords[:, 0]  # [E]
            d_lon = delta_coords[:, 1]  # [E]

            # Euclidean distance with epsilon to avoid division by zero
            dist_sq = torch.clamp(d_lat ** 2 + d_lon ** 2, min=1e-5)
            dist = torch.sqrt(dist_sq)

            # Directional unit vectors along edges
            dir_lat = d_lat / dist  # [E] (North component)
            dir_lon = d_lon / dist  # [E] (East component)

            # Projected directional temperature gradient along edge: ∂T/∂s
            edge_grad_t = delta_temp / dist.unsqueeze(0)  # [B, E]

            # Projected velocity along edge: u · e_ij = u * dir_lon + v * dir_lat
            u_src = u_velocity[:, src]  # [B, E]
            v_src = v_velocity[:, src]  # [B, E]
            v_projected = u_src * dir_lon.unsqueeze(0) + v_src * dir_lat.unsqueeze(0)  # [B, E]

            # Edge advective flux contribution: (u · e_ij) * (∂T/∂s)
            edge_advection = v_projected * edge_grad_t  # [B, E]

            # Edge diffusion flux contribution: (T_j - T_i) / dist²
            edge_diffusion = delta_temp / dist_sq.unsqueeze(0)  # [B, E]
        else:
            # Default unweighted graph Laplacian & uniform distance
            edge_advection = delta_temp * (u_velocity[:, src] + v_velocity[:, src])
            edge_diffusion = delta_temp

        # Aggregate outgoing edge contributions per source node
        advection = torch.zeros((B, N), device=temperature.device, dtype=temperature.dtype)
        diffusion = torch.zeros((B, N), device=temperature.device, dtype=temperature.dtype)
        node_degrees = torch.zeros(N, device=temperature.device, dtype=temperature.dtype)

        # Count node out-degrees
        node_degrees.scatter_add_(0, src, torch.ones(E, device=temperature.device, dtype=temperature.dtype))
        node_degrees = torch.clamp(node_degrees, min=1.0).unsqueeze(0)  # [1, N]

        # Scatter add edge contributions into nodes
        src_expanded = src.unsqueeze(0).expand(B, -1)  # [B, E]
        advection.scatter_add_(1, src_expanded, edge_advection)
        diffusion.scatter_add_(1, src_expanded, edge_diffusion)

        # Normalize by node neighborhood degree
        advection = advection / node_degrees
        diffusion = diffusion / node_degrees

        return advection, diffusion

    def forward(
        self,
        pred_sst_next: torch.Tensor,
        x_features: torch.Tensor,
        edge_index: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculates the physical advection-diffusion residual loss:
            Loss_phys = || (T_pred - T_t)/Δt + (u·∇)T - κ ∇²T ||²

        Parameters
        ----------
        pred_sst_next : torch.Tensor
            [B, N] Predicted SST at t+1
        x_features : torch.Tensor
            [B, T, N, F] or [B, N, F] feature tensor.
            Extracts latest timestep if 4D tensor is passed.
        edge_index : torch.Tensor
            [2, E] Graph edge index
        coords : torch.Tensor, optional
            [N, 2] Node spatial coordinates

        Returns
        -------
        torch.Tensor
            Scalar physics residual loss
        """
        # If full temporal history is provided [B, T, N, F], extract latest timestep t
        if x_features.ndim == 4:
            x_t = x_features[:, -1, :, :]  # [B, N, F]
        else:
            x_t = x_features  # [B, N, F]

        # Extract ocean state variables at current time step t
        temp_t = x_t[:, :, self.sst_idx]        # [B, N]
        u_t = x_t[:, :, self.u_idx]             # [B, N]
        v_t = x_t[:, :, self.v_idx]             # [B, N]

        # Extract coordinates from features if not passed explicitly
        if coords is None and x_t.shape[-1] > max(self.lat_idx, self.lon_idx):
            lat_t = x_t[0, :, self.lat_idx]
            lon_t = x_t[0, :, self.lon_idx]
            coords = torch.stack([lat_t, lon_t], dim=-1)  # [N, 2]

        # 1. Temporal derivative: ∂T/∂t ≈ (T_{t+1} - T_t) / Δt
        temporal_derivative = (pred_sst_next - temp_t) / self.delta_t  # [B, N]

        # 2. Spatial Advection & Diffusion terms on the graph
        advection, diffusion = self.compute_graph_spatial_derivatives(
            temperature=temp_t,
            u_velocity=u_t,
            v_velocity=v_t,
            edge_index=edge_index,
            coords=coords,
        )

        # 3. Complete Thermal Conservation Residual:
        # R = ∂T/∂t + (u·∇)T - κ ∇²T
        physics_residual = temporal_derivative + advection - (self.diffusivity_kappa * diffusion)

        # 4. Compute Loss
        if self.reduction == "mean":
            return torch.mean(physics_residual ** 2)
        elif self.reduction == "sum":
            return torch.sum(physics_residual ** 2)
        elif self.reduction == "none":
            return physics_residual ** 2
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
