# Air Traffic Management: Neural Network Method

## Group Members
- Brandon Rutagamirwa  
- Zhen Wei Yap  
- Onur Kurt  
- Staines Rajith  
- Hardit Ravinder Saini  
- Osinachi Ogemena  

---

## Overview
This section of the coursework focuses on the design and implementation of a Neural Network–based regression model for predicting aircraft taxi time within an Air Traffic Management (ATM) context. Emphasis is placed on the neural network architecture, training process, and performance evaluation, using dimensionality-reduced inputs to assess the effectiveness of supervised learning for taxi time prediction.


---

## Model Structure

**Choice of Model:**  
Multi-Layer Perceptron (MLP) with Back Propagation Neural Network (BPNN)

### Architecture
- **Input Layer**
  - 2 neurons corresponding to principal components
  - Bias term included

- **Hidden Layer**
  - 1 hidden layer with 12 neurons
  - Bias term included
  - Activation function: Sigmoid

- **Output Layer**
  - 1 neuron
  - Activation function: Linear (identity), suitable for regression tasks

---

## Training Process

- **Learning Rate:** 0.001  
- **Epochs:** 1000  
- **Loss Function:** Mean Squared Error (MSE)

### Training Steps
1. **Forward Propagation**  
   Input data (`x_train`) is passed through the network to generate predicted taxi times.

2. **Loss Calculation**  
   The Mean Squared Error between predicted values and actual values (`y_train`) is computed.

3. **Back Propagation**  
   Gradients of the loss function with respect to network weights are calculated and used to update the model parameters.

---

## Dataset Splitting
The dataset is divided as follows:
- **Training Set:** 70%  
- **Testing Set:** 20%  
- **Validation Set:** 10%  

---

## Statistical Tests and Evaluation

### Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### Statistical Analysis
- Residual analysis comparing predicted versus actual taxi times to assess model reliability and error distribution.

---

## Conclusion
The implemented MLP model demonstrates the applicability of neural networks in Air Traffic Management prediction tasks. Performance metrics and residual analysis are used to validate accuracy and robustness.
