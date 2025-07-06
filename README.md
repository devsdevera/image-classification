# Image Classification Using MLP and CNN

This project is a deep learning-based image classification system developed as a capstone project. It includes the implementation and training of both a Multilayer Perceptron (MLP) and a Convolutional Neural Network (CNN) using PyTorch. The system is designed to classify images into one of three categories and supports training on synthetic and clean datasets with flexible preprocessing and training configurations.

## 📁 Project Structure

```
.
├── train.py              # Main training script
├── cleandata/            # Directory containing original labeled images
├── generated/            # (Optional) Directory containing synthetic training data
├── model.pth             # Saved trained CNN model
├── mlp.pth               # (Optional) Saved trained MLP model
└── README.md             # Project documentation
```

## 🧀 Models Implemented

### 1. **Multilayer Perceptron (MLP)**

* Three fully connected layers.
* Accepts flattened 300x300 RGB images.
* Used for initial testing or baseline.

### 2. **Convolutional Neural Network (CNN)**

* Four convolutional layers with ReLU and MaxPooling.
* Dropout layers for regularization.
* Fully connected layers leading to a 3-class softmax output.
* More robust and accurate than MLP for image tasks.

## 🔄 Training Process

* **Preprocessing**: Includes resizing, flipping, color jittering, normalization.
* **Dataset Splitting**: 80/20 train-test split. Supports training on:

  * Clean data only
  * Clean + Synthetic data
* **Optimization**:

  * MLP: SGD with momentum.
  * CNN: Adam with OneCycleLR scheduler.

## ✅ Evaluation

The script includes a detailed evaluation function:

* Computes per-class and overall accuracy.
* Helps visualize model performance across classes.

## 🤦‍♂️ Usage

1. **Prepare the Dataset**:

   * Place your clean labeled images in `./cleandata/`.
   * Optional: Place your synthetic/generated data in `./generated/`.

2. **Train the CNN Model**:

   ```bash
   python train.py
   ```

3. **Enable/Disable MLP or CNN Training**:
   Uncomment respective lines in `train.py` to train the MLP or evaluate saved models.

4. **Model Saving**:

   * CNN is saved as `model.pth`
   * MLP can be saved as `mlp.pth` (if training is enabled)

## 🥪 Dependencies

Make sure to install the following Python packages:

```bash
pip install torch torchvision numpy
```

## 🖼 Image Specifications

* Input size: `300x300`
* Channels: `RGB`
* Classes: 3 (automatically inferred from `cleandata` subdirectories)

## ⚙️ Configuration Flags (in `train.py`)

| Flag         | Purpose                                  |
| ------------ | ---------------------------------------- |
| `validation` | Prints dataset sizes and evaluates model |
| `synthetic`  | Toggles use of synthetic/generated data  |

## 📊 Sample Output

```
cuda
train dataset size 1800
Classes: ['class1', 'class2', 'class3']
Epoch 1/10, Loss: 1.2345, Accuracy: 65.42%
...
CNN model saved successfully
CNN total train time: 120.45s
```

## 👤 Author

**Deveremma**
Student ID: `300602434`
