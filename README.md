# [Generalizing A Person Retrieval Model Hetero- and Homogeneously](http://openaccess.thecvf.com/content_ECCV_2018/html/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.html)
================================================================

Code for Generalizing A Person Retrieval Model Hetero- and Homogeneously (ECCV 2018). [[paper]](http://openaccess.thecvf.com/content_ECCV_2018/html/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.html) 

### Preparation

#### Requirements: Python=3.6 and Pytorch=0.4.0

1. Install [Pytorch](http://pytorch.org/)

2. Download dataset
   
   - reid_dataset [[GoogleDriver]](https://drive.google.com/open?id=1GABeDHWOEBGhEceYiSwr3ghUAl0Ne12X)
   
   - The reid_dataset including Market-1501 (with CamStyle), DukeMTMC-reID (with CamStyle), and CUHK03
   
   - Unzip reid_dataset under 'HHL/data/'
   
### CamStyle Generation
You can train CamStyle model and generate CamStyle imgaes with [stargan4reid](https://github.com/zhunzhong07/HHL/tree/master/stargan4reid)

### Training and test domain adaptation model for person re-ID

1. Baseline
  ```Shell
  # For Duke to Market-1501
  python baseline.py -s duke -t market --logs-dir logs/duke2market-baseline
  # For Market-1501 to Duke
  python baseline.py -s market -t duke --logs-dir logs/market2duke-baseline
  ```

2. HHL
  ```Shell
  # For Duke to Market-1501
  python HHL.py -s duke -t market --logs-dir logs/duke2market-HHL
  # For Market-1501 to Duke
  python HHL.py -s market -t duke --logs-dir logs/market2duke-HHL
  ```
  

### Results

<table>
   <tr>
      <td></td>
      <td colspan="2">Duke to Market</td>
      <td colspan="2">Market to Duke</td>
   </tr>
   <tr>
      <td>Methods</td>
      <td>Rank-1</td>
      <td>mAP</td>
      <td>Rank-1</td>
      <td>mAP</td>
   </tr>
   <tr>
      <td>Baseline</td>
      <td>44.6</td>
      <td>20.6</td>
      <td>32.9</td>
      <td>16.9</td>
   </tr>
   <tr>
      <td>HHL</td>
      <td>62.2</td>
      <td>31.4</td>
      <td>46.9</td>
      <td>27.2</td>
   </tr>
</table>


### References

- [1] Our code is conducted based on [open-reid](https://github.com/Cysu/open-reid)

- [2] StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation
, CVPR 2018

- [3] Camera Style Adaptation for Person Re-identification. CVPR 2018.


### Citation

If you find this code useful in your research, please consider citing:

    @inproceedings{zhong2018generalizing,
    title={Generalizing A Person Retrieval Model Hetero- and Homogeneously},
    author={Zhong, Zhun and Zheng, Liang and Li, Shaozi and Yang, Yi},
    booktitle ={ECCV},
    year={2018}
    }

    
### Contact me

If you have any questions about this code, please do not hesitate to contact me.

[Zhun Zhong](http://zhunzhong.site)
