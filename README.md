# Exercise 2.4: Channel GAN Implementation (Rayleigh Fading)

本專案旨在利用 **條件生成對抗網路 (Conditional GAN, CGAN)** 在不具備顯式通道狀態資訊 (CSI) 的情況下，模擬並學習 Rayleigh 衰落通道的數據分佈。

## 專案內容
- **目標**：透過 CGAN 學習通道分佈，根據輸入條件（發射符號與通道係數）生成模擬的接收訊號。
- **核心技術**：
    - **QuaDRiGa**：用於生成符合 3GPP 38.901 UMi NLOS 標準的真實通道數據集 (`.mat`)。
    - **CGAN (WGAN-GP)**：利用 Wasserstein GAN 結合 Gradient Penalty 架構進行訓練，確保訓練穩定性。
    - **Conditional Vector**：以 $[Re(x), Im(x), Re(h), Im(h)]$ 作為條件，指導生成器 (Generator) 產生接收訊號 $y$。

## 環境配置 (Windows/RTX 4050)
本專案運行於 Windows 環境，並使用 NVIDIA RTX 4050 進行 GPU 加速：
1. **虛擬環境**：`conda create -n cg_env python=3.9`
2. **CUDA 支援**：`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
3. **必要依賴**：
   ```bash
   pip install tensorflow==2.10.1 numpy==1.24.4 scipy==1.13.1 matplotlib==3.9.1 protobuf==3.19.6