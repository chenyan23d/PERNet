# PERNet-Enhanced Camouflaged Object Detection: A Perception-Enhancement-Refinement Framework

## 1. Repository

    *   This repository provides the official implementation of "***Enhanced Camouflaged Object Detection: A Perception-Enhancement-Refinement Framework***" (Currently submitted to The Visual Computer).
    *   The code implements the proposed Perception-Enhancement-Refinement framework. If this work contributes to your research, we would appreciate a citation to our manuscript.
  
## 2. Environment Configuration
    *   run: pip install -r requirements.txt
## 3. Data & Pre-trained Models Preparation
### Download Datasets
    *   Download the camouflaged object detection datasets and move it into `./data/` directory.
### Download Model Weights
    *   PERNet Model: Download from https://pan.baidu.com/s/1bjoYE5Mn3ZgixubZVQik_g?pwd=m5j2 and move it into `./PERNet.pth`.
    *   PVT_v2 Backbone:Download pvt_v2_b4 weights and move it into `./pvt_v2_b4.pth`.
## 4. Training
    *   After preparing the training dataset: run python train.py
## 5. Testing & Inference
    *   To generate prediction maps using pre-trained models: run python infer.py
    *   You can also download prediction maps ('CAMO', 'COD10K', 'NC4K') from https://pan.baidu.com/s/1CTIqN2b5Jte5zynCTzIh_A?pwd=w7dc.
