# Deep Learning Framework from Scratch 

A fully dynamic, Object-Oriented Deep Learning framework built entirely from scratch using only **NumPy**. This project demonstrates the inner workings of Neural Networks, including forward propagation, backpropagation, and mathematical optimization, without relying on high-level libraries like TensorFlow or PyTorch.

To validate the framework's architecture, it was trained and tested on the **NASA Kepler Exoplanet Search dataset** to classify celestial objects as either *Confirmed Exoplanets* or *False Positives*.

### Key Features

*   **Dynamic Architecture:** Supports instantiation of $N$ hidden layers dynamically through a custom Object-Oriented loop.
*   **Backpropagation Engine:** Custom-built backward pass using negative indexing to traverse matrices and calculate gradients correctly across multiple layers.
*   **He Initialization:** Implemented *Kaiming He Initialization* (`np.sqrt(2.0 / n_inputs)`) for Dense Layers to prevent the "Dying ReLU" problem and Vanishing Gradients in deep networks.
*   **Memory Isolation:** Each layer maintains its independent activation instances, preserving local inputs for accurate derivative calculations.
*   **Evaluation Metrics:** Built-in accuracy tracking, Categorical Cross-Entropy Loss visualization, and Confusion Matrix generation.

### Tech Stack & Components

*   **Language:** Python
*   **Core Engine:** NumPy (Matrix multiplication, derivatives, mathematical transformations)
*   **Data Processing:** Pandas, Scikit-Learn (Z-Score normalization, One-Hot Encoding, Train/Test split)
*   **Visualization:** Matplotlib (Epoch vs. Loss tracking)

### Implemented Classes:
*   `Layer_Dense`: Weights, biases, forward dot products, and gradient routing.
*   `Activation_ReLU`: Forward pass and derivative mask.
*   `Activation_Softmax`: Exponentiation with overflow protection and normalization.
*   `Loss_CategoricalCrossEntropy`: Negative log-likelihood calculation.
*   `Optimizer_SGD`: Stochastic Gradient Descent parameter updates.
*   `Simple_Training_Loop`: The orchestrator that dynamically links layers, epochs, and batches.

### Dataset: NASA Kepler Exoplanet Search

The framework is tested against the cumulative Kepler dataset. It processes physical features of celestial objects to predict their nature:
*   **Features Used:** Transit duration, depth, stellar effective temperature (steff), model SNR, planetary radius (prad), and orbital period.
*   **Preprocessing:** Applied Z-Score normalization and One-Hot Encoding `[1, 0]` for binary classification.

## How to Run

1. #### Clone this repository:
   ```bash
   git clone [https://github.com/Ryan-Fidelis/mlp-mini-framework.git](https://github.com/Ryan-Fidelis/mlp-mini-framework.git)
   ```
   
2. #### Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
   
3. #### Download the Kepler cumulative.csv dataset from Kaggle or NASA and place it in the root directory.
  
4. #### Run the training and validation pipeline:
   ```bash
   python training.py
   ```
   
## Outputs
Upon running, the script will:

1. Train the network for the specified number of epochs.

2. Display a Matplotlib graph showing the Loss curve decreasing over time.

3. Output the Test Loss, Test Accuracy, and a Pandas-generated Confusion Matrix on unseen data in the terminal.
