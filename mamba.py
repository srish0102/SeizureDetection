import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    """
    A Pure PyTorch implementation of the Mamba Selective State Space Model block.
    This implementation is 100% compatible with Windows, PyTorch 2.0+, and is designed
    to run natively on CUDA GPUs (like the RTX 5050) without custom C++/CUDA kernels.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Projected input paths
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # 1D Convolutional path for local temporal context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation function
        self.act = nn.SiLU()

        # Selective State Space Model projection layers
        # Projection to step size delta (dt)
        self.x_proj = nn.Linear(self.d_inner, 1 + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize S4 parameters (S4D-style initialization)
        # S4D Real part initialization: A must be negative to guarantee stability
        A_log = torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(A_log)
        
        # D parameter represents the direct skip connection (residual feedthrough)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to the target model dimension
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, u):
        """
        Args:
            u: Tensor of shape (Batch, Time, Features) [B, L, D]
        Returns:
            Tensor of shape (Batch, Time, Features) [B, L, D]
        """
        B, L, _ = u.shape

        # Step 1: Linear projection of inputs into 2 branches
        projected = self.in_proj(u)  # [B, L, 2 * d_inner]
        x, res = projected.chunk(2, dim=-1)  # Split into branch x and residual branch res

        # Step 2: 1D Depthwise Convolution along the temporal length
        x = x.transpose(1, 2)  # [B, d_inner, L] for Conv1D
        x = self.conv1d(x)[:, :, :L]  # Apply convolution and slice off padding
        x = x.transpose(1, 2)  # [B, L, d_inner]
        x = self.act(x)

        # Step 3: Selective State Space Parameters (Selective Scan)
        ssm_params = self.x_proj(x)  # [B, L, 1 + 2 * d_state]
        dt, B_matrix, C_matrix = ssm_params.split([1, self.d_state, self.d_state], dim=-1)

        # Project dt into the internal state dimension size
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]
        
        # Step 4: Run Selective Scan Recurrence Loop
        # Retrieve A matrix (must be negative)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Execute the scan loop sequentially along the time dimension (L)
        y = self.selective_scan(x, dt, A, B_matrix, C_matrix, self.D)

        # Step 5: Multiplicative Gated connection with the residual branch (SiLU-gated)
        gated_output = y * self.act(res)

        # Step 6: Final linear projection back to d_model dimension
        return self.out_proj(gated_output)

    def selective_scan(self, u, dt, A, B, C, D):
        """
        Executes sequential discretization and state updates.
        
        dt: [B, L, d_inner]
        A:  [d_inner, d_state]
        B:  [B, L, d_state]
        C:  [B, L, d_state]
        D:  [d_inner]
        """
        B_size, L, d_inner = u.shape
        d_state = A.shape[1]
        device = u.device

        # Initialize hidden state h with zeros
        h = torch.zeros(B_size, d_inner, d_state, device=device)
        y = torch.zeros(B_size, L, d_inner, device=device)

        # Precompute matrix exponents for the continuous-to-discrete A transition (At)
        # Using zero-order hold (ZOH) discretization: At = exp(dt * A)
        # dt is [B, L, d_inner, 1], A is [1, 1, d_inner, d_state]
        dt_expanded = dt.unsqueeze(-1)  # [B, L, d_inner, 1]
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # [1, 1, d_inner, d_state]
        At = torch.exp(dt_expanded * A_expanded)  # [B, L, d_inner, d_state]

        # Discrete B transition: Bt = dt * B
        # dt_expanded is [B, L, d_inner, 1], B is [B, L, 1, d_state]
        B_expanded = B.unsqueeze(2)  # [B, L, 1, d_state]
        Bt = dt_expanded * B_expanded  # [B, L, d_inner, d_state]

        # Scan recurrence loop over time (sequential dependency modeling)
        for t in range(L):
            # Input signal at step t
            u_t = u[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]
            
            # Update Hidden State: h_t = At * h_{t-1} + Bt * u_t
            h = At[:, t] * h + Bt[:, t] * u_t  # [B, d_inner, d_state]
            
            # Compute output state projection: y_t = C_t * h_t
            # C_t is [B, d_state], h is [B, d_inner, d_state]
            C_t = C[:, t].unsqueeze(1)  # [B, 1, d_state]
            y_t = torch.sum(h * C_t, dim=-1)  # [B, d_inner]
            
            y[:, t, :] = y_t

        # Apply Feedthrough bypass (Skip connection D)
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        return y
