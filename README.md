# Cortex-Net: Privacy-Preserving Federated Learning for Brain Tumor Detection

Cortex-Net is a simulated, end-to-end **Federated Learning (FL)** platform equipped with **Local Differential Privacy (LDP)** designed for collaborative brain tumor classification from MRI slices. 

This platform allows separate, non-cooperating medical institutions (Hospital A, B, and C) to collaboratively train a global **Convolutional Neural Network (CNN)** on their respective MRI slices without centralizing or sharing raw patient data.

---

## 🚀 Key Features

* **Federated Learning (FedAvg)**: Collaborative decentralized training. Parameters are trained locally on client silos and aggregated at the server using the Federated Averaging algorithm.
* **Local Differential Privacy (LDP)**: Protects client models against membership inference and reconstruction attacks. Implements local gradient update clipping ($L_2$ norm) and Gaussian noise addition before weight transmission.
* **Synthetic MRI Engine**: A built-in procedural image generator that synthesizes realistic $128 \times 128$ grayscale 2D brain MRI slices (Normal vs. Tumor) with ventricles, gyri texture, tumors, surrounding edema swelling, and scanner noise.
* **Glassmorphic Web Dashboard**: A premium, real-time UI showing:
  - Slide configurations for rounds, epochs, learning rate, and DP settings.
  - Interactive network topology animations showing parameter uploads and downloads.
  - Telemetry charts plotting Global Test Accuracy and Training Loss.
  - Diagnostic Sandbox: Upload custom MRI files or generate random slices on-the-fly to test model predictions with confidence dials and scan-beam effects.
* **Automated Unit Testing**: Includes unit tests verifying the model outputs, image generator parameters, and DP clipping algorithms.

---

## 🛠️ Technology Stack

* **Deep Learning**: PyTorch (`torch`, `torchvision`)
* **Backend Server**: FastAPI (Python), Uvicorn, WebSockets
* **Data Processing**: NumPy, Pillow, Scikit-Learn
* **Frontend UI**: Vanilla HTML5, CSS3 (Glassmorphism theme), JavaScript
* **Telemetry Visualizations**: Chart.js

---

## 📂 Project Structure

```text
federated_brain_tumor_detection/
├── static/
│   ├── index.html       # Web dashboard layout
│   ├── style.css        # Premium glassmorphic dark-mode CSS
│   └── app.js           # Websocket receiver, topology animator, & Chart.js plots
├── synthetic_data.py    # Procedural MRI generation & IID/Non-IID partitioning
├── model.py             # CNN model & Local Differential Privacy clipping
├── federated.py         # Federated Coordinator (FedAvg orchestrator)
├── server.py            # FastAPI REST & Websocket server routes
├── main.py              # Application entry point (runs uvicorn & starts browser)
├── test_fl.py           # Verification unit tests
├── requirements.txt     # Python package requirements
└── .gitignore           # Ignores local dataset folders and caches
```

---

## ⚙️ Setup and Running Instructions

### 1. Clone the Repository & Install Dependencies
First, clone the repository to your local machine and install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
Confirm everything is set up correctly by executing the verification test suite:
```bash
python test_fl.py
```

### 3. Start the Platform
Launch the coordinator server and the web dashboard by running `main.py`:
```bash
python main.py
```
This starts the FastAPI server on `http://127.0.0.1:8000` and automatically opens a new tab in your web browser.

---

## 🧠 Core Mathematical Concepts

### Federated Averaging (FedAvg)
Rather than centralizing patient datasets $D_1, D_2, \dots, D_K$ on a single server, each client trains a local copy of the model weights $W$ on its own data for $E$ epochs. At the end of each round, the central coordinator aggregates the weights:
$$W_{\text{global}}^{t+1} = \sum_{k=1}^K \frac{n_k}{N} W_k^{t+1}$$
Where $n_k$ is the local sample size of client $k$, and $N$ is the total samples across all active clients.

### Local Differential Privacy (LDP)
To ensure that the uploaded parameters $W_k$ do not leak patient MRI features, each hospital clips and noises their weight updates ($\Delta W = W_{\text{local}} - W_{\text{global}}$):
1. **Clipping**: Bounds parameter update sensitivity to a threshold $C$:
   $$\Delta W_{\text{clipped}} = \Delta W \cdot \min\left(1, \frac{C}{\|\Delta W\|_2}\right)$$
2. **Noising**: Adds calibrated Gaussian noise scaled by the clip bound and noise multiplier $\sigma$:
   $$\Delta W_{\text{noised}} = \Delta W_{\text{clipped}} + \mathcal{N}(0, \sigma^2 C^2)$$
3. **Reassembly**: Sends $W_{\text{global}} + \Delta W_{\text{noised}}$ to the coordinator.
