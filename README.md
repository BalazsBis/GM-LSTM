#Model: Cuda_MDN_LSTM

This model combines a Long Short-Term Memory (LSTM) network with a Mixture Density Network (MDN) to model the dynamics and variability of in-situ soil moisture data.

Device Configuration:

    - The model can be run either on a GPU or CPU based on the running_device configuration.
    - If the model is running on a GPU, it utilizes cuda:0, and the GPU device name is printed.
    - If the model is running on a CPU, it defaults to the "cpu" device.

Architecture:

The LSTM is designed to capture temporal dependencies in soil moisture time series.

The model includes an LSTM layer with the following parameters:

    - Input size (input_size): The number of features in the input.
    - Hidden size (hidden_size): The number of hidden units in the LSTM.
    - Number of layers (no_of_layers): The number of stacked LSTM layers.

Gaussian Mixture Model (MDN): The model outputs a mixture of Gaussians to represent variability in soil moisture predictions:

    - Mixing coefficients (pi): The weights for each Gaussian component.
    - Means (mu): The means of the Gaussian components.
    - Standard deviations (sigma): The standard deviations of the Gaussian components.


Loss Function: nll_loss (Negative Log-Likelihood Loss)
T
he Negative Log-Likelihood (NLL) loss function is used to train the Mixture Density Network (MDN). The goal is to minimize the negative log-likelihood, which maximizes the likelihood of the observed data under the predicted mixture distribution.

Steps of the Loss Function:

    - Softmax Normalization: The mixing coefficients (pi) are normalized using the softmax function to ensure they are non-negative and sum to 1 across components for each data point.
    - Gaussian Distribution Calculation: For each data point, the Gaussian probability density function (PDF) is calculated using the predicted means (mu) and standard deviations (sigma) for each Gaussian component.
    - Logarithm of the PDF: The log of the Gaussian PDF is computed to ensure numerical stability.
    - Log-Sum-Exp: The log of the sum of exponentiated log-probabilities is computed for numerical stability, ensuring that small probabilities from different components are added without causing overflow or underflow errors.
    - Final Loss Calculation: The mean of the negative log-likelihood is computed across the entire batch, which is used as the loss to optimize the model.
