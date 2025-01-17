import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import Adadelta
from torch.nn.functional import mse_loss
import numpy as np
import logging
import inspect

logger = logging.getLogger(__name__)

class BarlowTwinsLoss(nn.Module):

    def __init__(self, lamb: float):
        super(BarlowTwinsLoss, self).__init__()
        self.lamb = lamb

    def forward(self, z_a: Tensor, z_b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # normalize repr. along the batch dimension
        z_a_norm, z_b_norm = self._normalize(z_a, z_b)
        batch_size: int = z_a.size(0)

        # cross-correlation matrix
        c = z_a_norm.T @ z_b_norm
        c[c.isnan()] = 0.0
        valid_c = c[~c.isinf()]
        limit = 1e+30 if c.dtype == torch.float32 else 1e+4
        try:
            max_value = torch.max(valid_c)
        except RuntimeError:
            max_value = limit # type: ignore
        try:
            min_value = torch.min(valid_c)
        except RuntimeError:
            min_value = -limit # type: ignore
        c[c == float("Inf")] = max_value if max_value != 0.0 else limit
        c[c == float("-Inf")] = min_value if min_value != 0.0 else -limit
        c.div_(batch_size)

        invariance_loss = torch.diagonal(c).add_(-1).pow_(2).sum()
        redundancy_reduction_loss = self._off_diagonal(c).pow_(2).sum()
        loss: Tensor = invariance_loss + self.lamb * redundancy_reduction_loss
        return loss, invariance_loss, redundancy_reduction_loss


    def _normalize(self, z_a: Tensor, z_b: Tensor) -> tuple[Tensor, Tensor]:
        """Helper function to normalize tensors along the batch dimension."""
        combined = torch.stack([z_a, z_b], dim=0)  # Shape: 2 x N x D
        normalized = F.batch_norm(
            combined.flatten(0, 1),
            running_mean=None,
            running_var=None,
            weight=None,
            bias=None,
            training=True,
        ).view_as(combined)
        return normalized[0], normalized[1]


    def _off_diagonal(self, x: Tensor) -> Tensor:
        # return a flattened view of the off-diagonal elements of a square matrix
        n: int
        m: int
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()



class MyCustomLoss(nn.Module):
    def __init__(self, hyperparameters_config):
        super(MyCustomLoss, self).__init__()  # Initialize nn.Module properly
        # You can define additional custom attributes here, if needed
        self.attention_max = hyperparameters_config.attention_max
        self.lambda_l2_reg = hyperparameters_config.lambda_l2_reg
        self.alpha_smooth_reg = hyperparameters_config.alpha_smooth_reg
        
    def forward(self, input, target, model):
        #print(f"This function was called by: {inspect.stack()}")
        input = input.unsqueeze(1)
        target = target.unsqueeze(1)

        #mse = mse_loss(input, target)
        mae = nn.L1Loss(reduction='sum')
        mse = mae(input, target)
        #print(f"mae: {mse}")

        max_training_output = torch.max(input)
        max_output = torch.max(target)
        peak_difference = self.attention_max * torch.abs(max_training_output - max_output)

        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        l2_reg = self.lambda_l2_reg * l2_norm 


        #smoothness = 0
        #for index in range(input.shape[0]):
        #    
        #    batch_tens = input[index, 0, :]
        #    batch_tens_smoothness = sum(torch.abs(batch_tens[i] - batch_tens[i - 1]) for i in range(1, batch_tens.shape[0]))
        #    smoothness += batch_tens_smoothness
        diffs = torch.abs(input[:, 0, 1:] - input[:, 0, :-1])
        smoothness = diffs.sum()
        
        smoothness = self.alpha_smooth_reg * smoothness 
        total = mse + l2_reg + smoothness + peak_difference

        #let's print the loss and the components, as percentages
        logger.info(f"FITNESS LOSS: mse: {mse}")
        logger.info(f"FITNESS LOSS: l2_reg: {l2_reg}")
        logger.info(f"FITNESS LOSS: smoothness: {smoothness}")
        logger.info(f"FITNESS LOSS: peak_difference: {peak_difference}")
        logger.info(f"FITNESS LOSS: total: {total}")
        logger.info(f"FITNESS LOSS: percentage mse: {mse/total}, percentage l2_reg: {l2_reg/total}, percentage smoothness: {smoothness/total}")
        return total

class MyCustomMSE(nn.Module):
    def __init__(self, hyperparameters_config):
        super(MyCustomMSE, self).__init__()  # Initialize nn.Module properly
        # You can define additional custom attributes here, if needed
        self.attention_max = hyperparameters_config.attention_max
        self.lambda_l2_reg = hyperparameters_config.lambda_l2_reg
        self.alpha_smooth_reg = hyperparameters_config.alpha_smooth_reg
        
    def forward(self, input, target, model):
        input = input.unsqueeze(1)
        target = target.unsqueeze(1)
        mae = nn.L1Loss(reduction='sum')
        mse = mae(input, target)
        
        print(f"MSE LOSS: mse: {mse}")

        return mse 

