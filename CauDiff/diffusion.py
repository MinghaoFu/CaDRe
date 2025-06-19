import torch
import torch.nn as nn
import numpy as np

class TimeDiffusion:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        """
        Initialize the diffusion model parameters
        n_steps: number of diffusion steps
        beta_start, beta_end: noise schedule parameters
        """
        self.n_steps = n_steps
        
        # Define noise schedule (linear beta schedule)
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        Neural network for denoising
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, t):
        # t is the diffusion step
        return self.net(x)

class DiffusionModel:
    def __init__(self, input_dim, n_steps=1000):
        self.input_dim = input_dim
        self.diffusion = TimeDiffusion(n_steps=n_steps)
        self.model = TimeSeriesModel(input_dim)
        
    def forward_diffusion(self, x_0, t):
        """
        Forward process - add noise to the input
        x_0: initial time series data
        t: diffusion step
        """
        alpha_hat = self.diffusion.alpha_hat[t]
        
        # Generate random noise
        epsilon = torch.randn_like(x_0)
        
        # Add noise according to diffusion schedule
        x_t = torch.sqrt(alpha_hat) * x_0 + torch.sqrt(1 - alpha_hat) * epsilon
        
        return x_t, epsilon
    
    def train_step(self, x_0, optimizer):
        """
        Single training step
        """
        # Sample random diffusion step
        t = torch.randint(0, self.diffusion.n_steps, (x_0.shape[0],))
        
        # Forward process
        x_t, epsilon = self.forward_diffusion(x_0, t)
        
        # Predict noise
        epsilon_pred = self.model(x_t, t)
        
        # Calculate loss
        loss = nn.MSELoss()(epsilon_pred, epsilon)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, n_samples, device="cpu"):
        """
        Generate new time series by denoising
        """
        # Start from random noise
        x = torch.randn(n_samples, self.input_dim).to(device)
        
        # Gradually denoise
        for t in reversed(range(self.diffusion.n_steps)):
            t_tensor = torch.full((n_samples,), t, device=device)
            
            # Predict noise
            epsilon_theta = self.model(x, t_tensor)
            
            alpha = self.diffusion.alpha[t]
            alpha_hat = self.diffusion.alpha_hat[t]
            
            # Remove noise step by step
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            x = (1 / torch.sqrt(alpha)) * (
                x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * epsilon_theta
            ) + torch.sqrt(1 - alpha) * noise
            
        return x

# Example usage:
"""
# Initialize model
input_dim = 24  # for 24 time steps
model = DiffusionModel(input_dim)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    loss = model.train_step(time_series_data, optimizer)
    
# Generate new time series
samples = model.sample(n_samples=10)
"""
