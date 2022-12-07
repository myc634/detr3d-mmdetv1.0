# DETR3D for mmdet3d-v1.0.0rc5 version

This repo contains the implementations of DETR3D (https://arxiv.org/abs/2110.06922). The original DETR3D is released based on mmdet3d-0.17.0 version, however, the mmdet3d reconfigured the coordinate system in version 1.0, which caused problems with some metrics such as mAOE. Refer to issue31(https://github.com/WangYueFt/detr3d/issues/31) we refactored this code and reproduced the results as in the paper.

### Prerequisite

1. mmcv (https://github.com/open-mmlab/mmcv)

2. mmdet (https://github.com/open-mmlab/mmdetection)

3. mmseg (https://github.com/open-mmlab/mmsegmentation)

4. mmdet3d (https://github.com/open-mmlab/mmdetection3d)

### Enviornment for mmdet3d and upper
      mmcv-full=1.6.0
      mmdet=2.24.0
      mmseg=0.20.0
      mmdet3d=1.0.0rc5
      pytorch=1.10.1

### Data
1. Follow the mmdet3d to process the data.

### Train
1. Downloads the [pretrained backbone weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) to pretrained/ 

2. For example, to train DETR3D on 8 GPUs, please use

`tools/dist_train.sh projects/configs/detr3d/detr3d_res101_gridmask.py 8`

### Evaluation using pretrained models
1. Download the weights accordingly.  

|  Backbone   | mAP | NDS | Download |
| :---------: | :----: |:----: | :------: |
|[DETR3D, ResNet101 w/ DCN](./projects/configs/detr3d/detr3d_res101_gridmask.py)|34.7|42.2|[model](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1uvrf42seV4XbWtir-2XjrdGUZ2Qbykid/view?usp=sharing)|
|[above, + CBGS](./projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py)|34.9|43.4|[model](https://drive.google.com/file/d/1sXPFiA18K9OMh48wkk9dF1MxvBDUCj2t/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1NJNggvFGqA423usKanqbsZVE_CzF4ltT/view?usp=sharing)|
|[DETR3D, VoVNet on trainval, evaluation on test set](./projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py)| 41.2 | 47.9 |[model](https://drive.google.com/file/d/1d5FaqoBdUH6dQC3hBKEZLcqbvWK0p9Zv/view?usp=sharing) &#124; [log](https://drive.google.com/file/d/1ONEMm_2W9MZAutjQk1UzaqRywz5PMk3p/view?usp=sharing)|




2. To test, use  
`tools/dist_test.sh /projects/configs/detr3d/detr3d_res101_gridmask.py /path/to/ckpt 8 --eval=bbox`



Explanation of the changes in the coordinate system caused by the mmdet3d version update:

link: https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/compatibility.md#v100rc0      



If you find this repo useful for your research, please consider citing the papers


```
@inproceedings{
   detr3d,
   title={DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries},
   author={Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle={The Conference on Robot Learning ({CoRL})},
   year={2021}
}
```
