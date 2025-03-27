# Cuda_GM_LSTM

This model combines a **Long Short-Term Memory (LSTM)** network with a **Mixture Density Network (MDN)** to model the **dynamics and variability of in-situ soil moisture data**.

---

## Device Configuration

- The model can be run either on a **GPU** or **CPU** based on the `running_device` configuration.
- If running on a **GPU**, it utilizes `cuda:0`, and the **GPU device name** is printed.
- If running on a **CPU**, it defaults to the `"cpu"` device.

---

## Architecture

### LSTM

The LSTM is designed to capture **temporal dependencies** in soil moisture time series.

**LSTM Layer Parameters:**

- `input_size`: Number of features in the input.
- `hidden_size`: Number of hidden units in the LSTM.
- `no_of_layers`: Number of stacked LSTM layers.

---

### Gaussian Mixture Model (MDN)

The model outputs a **mixture of Gaussians** to represent variability in soil moisture predictions:

- `pi`: Mixing coefficients — weights for each Gaussian component.
- `mu`: Means of the Gaussian components.
- `sigma`: Standard deviations of the Gaussian components.

---

## Loss Function: `nll_loss` (Negative Log-Likelihood Loss)

The **Negative Log-Likelihood (NLL)** loss function is used to train the Mixture Density Network (MDN).  
The goal is to **maximize the likelihood** of the observed data under the predicted mixture distribution by minimizing the NLL.

### Steps of the Loss Function:

1. **Softmax Normalization**  
   Mixing coefficients (`pi`) are normalized using the softmax function to ensure they are non-negative and sum to 1 across components for each data point.

2. **Gaussian Distribution Calculation**  
   For each data point, the Gaussian **Probability Density Function (PDF)** is calculated using the predicted `mu` and `sigma` values for each component.

3. **Logarithm of the PDF**  
   The **log of the Gaussian PDF** is computed to ensure numerical stability.

4. **Log-Sum-Exp**  
   The **log of the sum of exponentiated log-probabilities** is computed for numerical stability, ensuring small probabilities from different components are added without overflow/underflow errors.

5. **Final Loss Calculation**  
   The **mean of the negative log-likelihood** is computed across the entire batch, which is used as the **loss to optimize** the model.
