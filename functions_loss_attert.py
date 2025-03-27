import torch
import numpy as np
import pandas as pd
from typing import Dict

# Function for the Gaussian distribution
def gaussian_distribution(y, mu, sigma):
    """
    Compute the Gaussian probability density function values for given y, mu, and sigma.
    
    This function was taken from Neural Hydrology [#]_ and adapted for our specific case. 
    
    Parameters:
    y (Tensor): Target values.
    mu (Tensor): Means of the Gaussian components.
    sigma (Tensor): Standard deviations of the Gaussian components.
    
    Returns:
    Tensor: The probability density values for the Gaussian distribution.
    """
    
    # Avoid dividing by zero by adding a small epsilon to sigma
    sigma = sigma + 1e-8

    # Compute the exponent part: exp(-0.5 * ((y - mu) / sigma)^2)
    result = -0.5 * ((y - mu) / sigma)**2

    # Return the Gaussian PDF: exp(result) / (sigma * sqrt(2 * pi))
    return torch.exp(result) / (sigma * np.sqrt(2.0 * np.pi))

# Function for negative log likelihood
'''
Part 1: softmax function ensures that the mixing coefficients are non-negative and sum to 1 across the mixture components (for each data point)
Part 2: The result is the value of the Gaussian Probability Density Function for each component in the mixture, evaluated at the target value y. 
For each Gaussian component (with its own mu and sigma), the Gaussian PDF gives the likelihood of y under that Gaussian. The results will be a tensor
of shape (batch_size, num_components) where each value represents the probability density for a particular Gaussian component.
Part 3: We take the log of the Gaussian PDF to convert the probability densities into log-probabilities. Logarithms are useful because they convert
multiplications into additions, which are more numerically stable when dealing with small numbers. The small epsilon value is added to aviod taking
the log of zero, which is undefined and can lead to NaN errors. The addition ensures that result is never exactly zero.
Part 4: The torch.logsumexp function is a numerically stable way to compute the log of a sum of exponentials. This step ensures that adding small 
probabilities from different components does not lead to underflow or overflow of errors. 
Part 5: After applying logsumexp, log_sum represents the log-probability of the target y under the entire Gaussian Mixture Model. This is done for each
data point in the batch. The loss function is based on the negative-log likelihood (NLL). Minimizing the NNL means maximizing the likelihood of the data 
under the model. We compute the mean over all data points in the batch, which gives a scalar loss value. This value is used to guide the optimization
process.
'''
def nll_loss(pi, mu, sigma, y):
    """
    Compute the loss function for a Mixture Density Network (MDN).

    Parameters:
    pi (Tensor): Mixing coefficients for the Gaussian components (after softmax).
    mu (Tensor): Means of the Gaussian components.
    sigma (Tensor): Standard deviations of the Gaussian components.
    y (Tensor): Target values.

    Returns:
    Tensor: The mean negative log-likelihood loss for the MDN.
    """

    # Step 1: Ensure pi is normalized using softmax
    pi = torch.softmax(pi, dim=1)

    # Step 2: Compute the Gaussian probability density function
    result = gaussian_distribution(y, mu, sigma)

    # Step 3: Compute log of Gaussian PDF (to avoid numerical issues)
    log_result = torch.log(result + 1e-8)

    # Step 4: Multiply log-pdf by the mixture weights and use log-sum-exp for numerical stability
    log_sum = torch.logsumexp(torch.log(pi) + log_result, dim=1)

    # Step 5: Return the mean negative log-likelihood
    return -torch.mean(log_sum)