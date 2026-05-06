# Optimization Methods: SGD vs Adam on Unbalanced Datasets 🚀

**Author:** Francesco Marchini  
**Course:** Optimization Techniques for Machine Learning  
**Master's Degree Program:** Artificial Intelligence  

## 📖 Project Overview
This repository contains the implementation of a project work exploring the performance gap between Stochastic Gradient Descent (SGD) and adaptive methods (like Adam) in contexts with heavy-tailed class imbalances. 

Inspired by recent literature (Kunstner et al., 2024), this project isolates the complexities of language modeling by simulating heavily unbalanced two-class problems using **Logistic Regression**. The core objective is to empirical demonstrate how and why standard optimizers collapse towards the majority class, and how adaptive scaling combined with minibatching protects minority class updates.

## ✨ Implemented Optimizers
All algorithms have been implemented from scratch using Python and `numpy`, avoiding nested loops for tensor efficiency. The suite includes:

**Full Batch (FB) Methods:**
- Gradient Descent with Armijo Line Search
- Conjugate Gradient (Fletcher-Reeves) with Armijo Line Search and Restart

**Minibatch (MB) Methods:**
- Stochastic Gradient Descent with Polyak Stepsize

**Adaptive Methods (Supporting both FB and MB):**
- Adagrad
- RMSProp
- AdaDelta
- Adam

## 📂 Repository Structure

- `models/logistic_regression_model.py`: Model and loss definition
- `models/optimizers.py`: Containing all optimizer implementations
- `utils/metrics_and_baseline.py`: L-BFGS-B baseline, accuracy, and plotting tools
- `utils/armijo.py`: Armijo line search implementation
- `utils/dgp.py`: Synthetic data generation for unbalanced classification
- `main.py`: Main execution pipeline
- `results/`: Auto-generated execution outputs (logs and high-resolution plots)
- `README.md`: Project documentation

## ⚙️ Installation & Requirements
Ensure you have Python 3.8+ installed. The project relies on standard scientific libraries. You can install them via pip:

> pip install numpy matplotlib scipy scikit-learn

## 🚀 How to Run the Experiments
To execute the complete optimization pipeline, simply run the main script from the root directory:

> python main.py

### 📊 Understanding the Output (`/results`)
During execution, the script automatically tests all implemented optimizers across varying dataset sizes (N: 500, 2000, 5000) and unbalance ratios (p: 0.1, 0.01, 0.005). 

All outputs are automatically routed to the `results/` directory:
1. `experiment_log.txt`: Contains a detailed breakdown of Global Accuracy, Minority/Majority Class Accuracies, and Final Losses for every single configuration.
2. `Plots (.png)`: A 2x2 dashboard plot is generated for each run, showing the overall training loss, the overlapping majority/minority class losses, and isolated class loss graphs.

## 🔬 Key Findings
* **The Accuracy Paradox:** Global accuracy easily exceeds 99% in heavily unbalanced scenarios merely by ignoring the minority class. 
* **Adam's Advantage:** Adaptive methods (Adam, RMSProp) successfully protect rare feature gradients in the minibatch regime, maintaining significantly higher accuracy on the minority class compared to standard SGD.
* **Data Starvation Limits:** Under extreme deprivation (e.g., N=500, p=0.005 with only 8 positive samples), no optimization algorithm can reliably establish a decision boundary.

## 📚 References
1. F. Kunstner et al., *"Heavy-tailed class imbalance and why Adam outperforms gradient descent on language models."* NeurIPS (2024).
2. M. Lapucci, *Course Lecture Notes* (2025).
3. N. Loizou et al., *"Stochastic Polyak step-size for SGD: An adaptive learning rate for fast convergence."* AISTATS (2021).