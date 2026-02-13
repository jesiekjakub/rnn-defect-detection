# Time Series Defect Detection with RNN

<div align="center">

**Multi-label defect classification and root cause analysis for manufacturing sensor data**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.2+-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![RNN](https://img.shields.io/badge/RNN-Bi--directional%20LSTM-green.svg?style=for-the-badge)](https://pytorch.org/)
[![Attention](https://img.shields.io/badge/Attention-Per--Class-orange.svg?style=for-the-badge)](https://pytorch.org/)
[![Seq2Seq](https://img.shields.io/badge/Seq2Seq-Autoencoder-purple.svg?style=for-the-badge)](https://pytorch.org/)

</div>

---

Deep learning system for detecting multiple simultaneous defects in time-series data from 3 sensors. Identifies which defects are present and localizes the exact time intervals and sensors responsible.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/jesiekjakub/DL-Project3-RNN.git
cd DL-Project3-RNN

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

---

## 🏗️ Approaches

### Approach 1: Bi-directional LSTM with Per-Class Attention

Each defect type has its own attention mechanism, allowing the model to focus on different time windows independently for each defect class.

**Architecture:**
- Bi-directional LSTM encoder processes sequences in both forward and backward directions for full temporal context
- Separate attention layer for each defect type (5 independent attention mechanisms)
- Per-class binary classifiers produce individual defect probabilities
- Attention weights provide direct explainability by showing which timesteps were important for each defect

### Approach 2: Seq2Seq Autoencoder + Supervised Classifier

Three-stage pipeline combining unsupervised normality learning with supervised classification.

**Architecture:**
1. **Normality Learner:** Autoencoder trained exclusively on healthy samples learns the manifold of normal production behavior
2. **Feature Engineering:** Composite 9-channel features combining:
   - Original signal (3 channels)
   - Reconstruction residuals (3 channels) - highlights deviations from learned normality
   - Velocity/derivatives (3 channels) - captures rate of change and flatlines
3. **Supervised Classifier:** LSTM classifier operates on engineered features
4. **Explainability:** Region proposal network identifies candidate defect regions using statistical anomaly detection, verifies them with local classification, and applies consensus checking against global predictions

---

## 📊 Key Features

**Multi-Label Classification:** Detects multiple simultaneous defects occurring in a single sequence without specialized loss functions.

**Variable Length Handling:** Sequences vary from 40-60 timesteps. The system uses packed sequences where padding is applied to batch sequences to uniform length, but the LSTM is informed of actual sequence lengths so it ignores padded timesteps during computation. This prevents padding artifacts from affecting model predictions.

**Root Cause Analysis:**
- **Approach 1:** Attention weights directly reveal which timesteps the model considered important for each defect. Sensor importance is calculated by comparing variance in high-attention regions versus low-attention regions.
- **Approach 2:** Statistical analysis of residuals and velocity gradients proposes candidate time intervals. These candidates are verified by cropping and re-classifying them. Only regions where local predictions match global predictions are accepted as true defects.

**Synthetic Data Generation:** Base sine wave signals with configurable frequency and amplitude, injected with defect patterns (spikes, drops, flatlines, bumps, multi-sensor patterns), plus gaussian noise for realism.


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
