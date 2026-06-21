# MNIST Digit Recognition using CNN

A Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify handwritten digits from the classic MNIST dataset. The model processes gray-scale images of handwritten numbers (0-9) and achieves high classification accuracy.

## 🚀 Features
* **Framework:** TensorFlow 2.x & Keras
* **Dataset:** MNIST Handwritten Digits (60,000 training images, 10,000 testing images)
* **Architecture:** Sequential CNN with 2D Convolutional and Max Pooling layers
* **Data Pipeline:** Automatic 4D reshaping and float32 pixel normalization `[0.0, 1.0]`

---

## 📂 Project Structure
```text
MNIST_CNN/
│
├── .venv/                  # Virtual environment
├── .gitignore              # Files ignored by Git (e.g., venv, local models)
├── main.py                 # Core model architecture and training script
├── README.md               # Project documentation
└── mnist_cnn_model.keras   # Saved trained model weights
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd mnist-digit-recognition
   ```

2. **Activate the virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   * **Mac/Linux:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install tensorflow numpy
   ```

---

## 🧠 Model Architecture

The neural network is built using the Keras `Sequential` API with the following structure:
* **Convolutional Layer:** 32 filters, 3x3 kernel size, ReLU activation function. Expects an input shape of `(28, 28, 1)`.
* **Max Pooling Layer:** 2x2 pool size to downsample spatial dimensions.
* *(Note: Fully connected Dense layers and Softmax activation can be added to finalize classifications).*

---

## 📈 Usage

To load the dataset, preprocess the pixel arrays, build the architecture, and begin training the network, execute the main Python file:

```bash
python main.py
```
