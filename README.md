# PERNet-Enhanced Camouflaged Object Detection: A Perception-Enhancement-Refinement Framework

## 1. Repository

*   This repository provides code for "***Enhanced Camouflaged Object Detection: A Perception-Enhancement-Refinement Framework***" Currently submitted to The Visual Computer.
*   This code is directly linked to the aforementioned manuscript and implements the proposed Perception-Enhancement-Refinement framework. If our work contributes to your project, we would be grateful if you could cite our relevant manuscript.

## 2. Training/Testing

1.  To configure environment, run: pip install -r requirements.txt

2.  Downloading necessary data:

    *   downloading dataset and move it into `./data/`.

    *   downloading our weights and move it into `./PERNet.pth`，the pretrained model of our model can be downloaded at https://pan.baidu.com/s/1bjoYE5Mn3ZgixubZVQik_g?pwd=m5j2;

    *   downloading pvt_v2_b4 weights and move it into `./pvt_v2_b4.pth.pth`.

3.  Training Configuration:

    *   After you download training dataset, just run `train.py` to train our model.

4.  Testing Configuration:

    *   After you download all the pre-trained model and testing dataset, just run `infer.py` to generate the final prediction maps.

    *   You can also download prediction maps ('CHAMELEON', 'CAMO', 'COD10K') from .
