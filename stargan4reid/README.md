# Training CamStyle with StarGAN

CamStyle is trained with [StarGAN](https://github.com/yunjey/StarGAN)


### Preparation

#### Requirements: Python=3.6 and Pytorch=0.4.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset
   
   - reid_dataset [[GoogleDriver]](https://drive.google.com/open?id=1GABeDHWOEBGhEceYiSwr3ghUAl0Ne12X)
   
   - The reid_dataset including Market-1501 (with CamStyle), DukeMTMC-reID (with CamStyle), and CUHK03
   
   - Unzip reid_dataset under 'HHL/data/'

# Train and test CamStyle models

  ```Shell
  # For Market-1501
  sh train_test_market.sh
  # For Duke
  sh train_test_duke.sh
  ```

# Pre-trained CamStyle models
- Market [[Google Drive]](https://drive.google.com/open?id=1DGSpnHAq8y_HfmnPYZUsJrpDRkvLzNMe)

- Duke [[Google Drive]](https://drive.google.com/open?id=1etumVK_DdP0-_rkSQbU-XQlMLFT0MzAS)



## Citation
If you use this code for your research, please cite our papers.
```

@inproceedings{zhong2018generalizing,
title={Generalizing A Person Retrieval Model Hetero- and Homogeneously},
author={Zhong, Zhun and Zheng, Liang and Li, Shaozi and Yang, Yi},
booktitle ={ECCV},
year={2018}
}

@inproceedings{choi2018stargan,
 title = {StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},    
 author = {Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul},
 booktitle= {CVPR},
 Year = {2018}
}

```
