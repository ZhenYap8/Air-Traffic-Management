# Air Traffic Management: Neural Network Method

## Model Structure
**Choice of Model:** Multi-Layer Perceptron (MLP) with Back Propagation (BPNN)

*   **Input Layer:** Single layer with 2 neurons (corresponding to Principal Components) + Bias term.
*   **Hidden Layer:** Single layer with 12 neurons + Bias term.
    *   *Activation Function:* Sigmoid.
*   **Output Layer:** Single layer with 1 neuron.
    *   *Activation Function:* Linear (Identity) for regression.

## Training Process
*   **Learning Rate:** 0.001
*   **Epochs:** 1000
*   **Loss Metric:** Mean Squared Error (MSE)

The training process follows these steps:
1.  **Forward Propagation:** Passes `x_train` through all layers to produce the predicted taxi time.
2.  **Loss Function:** Calculates the difference between the predicted value and the actual `y_train`.
3.  **Back Propagation:** Computes the gradient of the loss with respect to all weights to update the model.

## Dataset Splitting
The dataset is divided into the following sub-datasets:
*   **Training:** 70%
*   **Testing:** 20%
*   **Validating:** 10%

## Statistical Tests & Evaluation
To evaluate the performance and reliability of the Neural Network, the following metrics and tests are conducted:
*   **Performance Metrics:**
    *   Mean Absolute Error (MAE)
    *   Root Mean Squared Error (RMSE)
    *   Coefficient of Determination ($R^2$)
*   **Statistical Analysis:** Analysis of Residuals (Predicted vs. Actual deviations).