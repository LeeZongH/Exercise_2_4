# Exercise 2.4: Channel GAN Implementation (Rayleigh Fading)

This project implements a **Conditional Generative Adversarial Network (CGAN)** to simulate and learn the distribution of Rayleigh fading channels without requiring explicit channel state information (CSI).

## Project Overview
- **Objective**: Learn the conditional probability distribution of the channel, allowing the generator to simulate received signals based on transmitted symbols and channel coefficients.
- **Key Technologies**:
    - **QuaDRiGa**: A versatile radio channel generator used to produce realistic channel datasets based on the 3GPP 38.901 UMi NLOS scenario (`.mat` format).
    - **CGAN (WGAN-GP)**: A Conditional GAN architecture integrated with Wasserstein GAN and Gradient Penalty (WGAN-GP) to ensure training stability and convergence.
    - **Conditioning Vector**: The model is conditioned on the transmitted signal (real/imag) and channel coefficients (real/imag), formulated as: $[Re(x), Im(x), Re(h), Im(h)]$.

## Environment Setup (Windows/RTX 4050)
The project is configured for Windows 10/11 with an NVIDIA RTX 4050 GPU for hardware acceleration:
1. **Create the virtual environment with the specific Python version:**
    ```bash
    conda create -n tensorflow python=3.9.19 -y

2. **Activate the environment:**
    ```bash
    conda activate tensorflow

3. **Install CUDA Toolkit via Conda:**
    ```bash
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

4. **Install all required packages via requirements.txt:**
   ```bash
   pip install -r requirements.txt