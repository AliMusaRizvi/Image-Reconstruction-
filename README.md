# Masked Autoencoder (MAE) for Self-Supervised Image Representation Learning

## Overview

This project implements a **Masked Autoencoder (MAE)** model for self-supervised image representation learning, based on the paper "Masked Autoencoders Are Scalable Vision Learners" by He et al. (2022). The implementation includes modern improvements from 2026, such as Flash Attention, RMSNorm, SwiGLU feed-forward networks, and advanced training techniques.

The project consists of:
- A Jupyter notebook (`mae-assignment02.ipynb`) that implements and trains the MAE model
- A Streamlit web app (`app.py`) for interactive image reconstruction demonstration
- Pre-trained model weights (`model_weights.pth`)

## What is MAE?

**Masked Autoencoders** are a self-supervised learning approach for vision tasks. The model:
1. **Encodes**: Divides input images into patches and randomly masks 75% of them
2. **Learns**: Trains an encoder-decoder architecture to reconstruct the masked patches
3. **Benefits**: Learns rich visual representations without requiring labeled data

### Key Features
- **Architecture**: ViT-Base/16 Encoder (~86M params) + ViT-Small/16 Decoder (~22M params)
- **Masking Ratio**: 75% of patches are masked, leaving only 25% visible
- **Dataset**: TinyImageNet (resized to 224×224)
- **Training**: 50 epochs with advanced optimization (AdamW, cosine warmup, AMP, etc.)

## Project Structure

```
├── mae-assignment02.ipynb    # Main implementation and training notebook
├── app.py                    # Streamlit web app for image reconstruction demo
├── model_weights.pth         # Pre-trained MAE model weights
└── venv/                     # Python virtual environment
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (for Flash Attention and torch.compile)
- CUDA-compatible GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy matplotlib pillow
pip install streamlit einops packaging
pip install jupyter notebook
```

### Virtual Environment

A virtual environment is already set up in the `venv/` directory. Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

## Usage

### Training the Model

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook mae-assignment02.ipynb
   ```

2. Run the cells in order:
   - **Environment Setup**: Verifies PyTorch, CUDA, and Flash Attention availability
   - **Configuration**: Sets hyperparameters and model architecture
   - **Dataset Preparation**: Loads and preprocesses TinyImageNet
   - **Model Components**: Defines Encoder, Decoder, and complete MAE
   - **Training Loop**: Trains the model with advanced techniques
   - **Evaluation**: Tests reconstruction quality and visualizes results

3. The notebook will save trained weights to `model_weights.pth`

### Running the Demo App

1. Ensure the model weights are available (`model_weights.pth`)

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to the provided URL (typically http://localhost:8501)

4. Upload an image and see the MAE model reconstruct masked regions

## Technical Details

### Model Architecture

#### Encoder (ViT-Base/16)
- **Input**: 224×224 images → 196 patches (16×16 each)
- **Dimensions**: 768 hidden, 12 layers, 12 attention heads
- **Parameters**: ~86M

#### Decoder (ViT-Small/16)
- **Input**: Encoded visible patches + learnable mask tokens
- **Dimensions**: 384 hidden, 12 layers, 6 attention heads
- **Parameters**: ~22M

### Training Configuration
- **Batch Size**: 64 per GPU (effective 128 with 2 GPUs)
- **Learning Rate**: 1.5e-4 (scaled by batch size)
- **Epochs**: 50 with 10 warmup epochs
- **Mask Ratio**: 75% patches masked
- **Optimization**: AdamW with decoupled weight decay, gradient clipping
- **Advanced Features**: Mixed precision (AMP), torch.compile, EMA, Flash Attention

### Loss Function
- **Primary**: Mean Squared Error (MSE) on reconstructed pixel values
- **Normalization**: Per-patch mean/std normalization for stable training
- **Auxiliary**: Frequency-domain loss (FFT L1) for texture preservation

## Results & Evaluation

The model is evaluated using:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **SSIM (Structural Similarity Index)**: Assesses structural preservation
- **Visual Inspection**: Qualitative assessment of reconstructed images

## Key Innovations (2026 Updates)

- **Flash Attention**: Efficient attention computation for long sequences
- **RMSNorm**: Improved normalization for better gradient flow
- **SwiGLU**: Advanced feed-forward networks
- **LayerScale**: Per-layer scaling for stable training
- **torch.compile**: JIT compilation for faster training
- **EMA**: Exponential moving average for model stabilization

## References

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) - He et al., 2022
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020

## Course Information

**Course**: Generative AI (AI4009)  
**Assignment**: No. 2  
**Semester**: Spring 2026  
**Institution**: National University of Computer and Emerging Sciences

## License

This project is for educational purposes as part of the Generative AI course assignment.