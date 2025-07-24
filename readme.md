<h1 align="center">[CVPR 2025] Rethinking Token Reduction with Parameter-Efficient Fine-Tuning in ViT for Pixel-Level Tasks</h1>

<div align="center">
  <hr>
  Cheng Lei, &nbsp;
  Ao Li, &nbsp;
  Hu Yao, &nbsp;
  Ce Zhu, &nbsp;
  Le Zhang, &nbsp;
  <br>
    University of Electronic Science and Technology of China. &nbsp;

  <h4>
    <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Lei_Rethinking_Token_Reduction_with_Parameter-Efficient_Fine-Tuning_in_ViT_for_Pixel-Level_CVPR_2025_paper.html">Paper</a> &nbsp; 
  </h4>
</div>

<blockquote>
<b>Abstract:</b> <i>Parameter-efficient fine-tuning (PEFT) adapts pre-trained models to new tasks by updating only a small subset of parameters, achieving efficiency but still facing significant inference costs driven by input token length. This challenge is even more pronounced in pixel-level tasks, which require longer input sequences compared to image-level tasks. Although token reduction (TR) techniques can help reduce computational demands, they often lead to homogeneous attention patterns that compromise performance in pixel-level scenarios. This study underscores the importance of maintaining attention diversity for these tasks and proposes to enhance attention diversity while ensuring the completeness of token sequences. Our approach effectively reduces the number of tokens processed within transformer blocks, improving computational efficiency without sacrificing performance on several pixel-level tasks. We also demonstrate the superior generalization capability of our proposed method compared to challenging baseline models. The source code will be made available at https://github.com/AVC2-UESTC/DAR-TR-PEFT.</i>
</blockquote>

<!-- <p align="center">
  <img width="1000" src="figs/framework.png">
</p> -->

---


## Install

For setup, refer to the [Quick Start](#quick-start) guide for a fast setup, or follow the detailed instructions below for a step-by-step configuration.

### Pytorch

The code requires `python>=3.9`, as well as `pytorch>=2.0.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### MMCV

Please install MMCV following the instructions [here](https://github.com/open-mmlab/mmcv/tree/master).

### xFormers

Please install xFormers following the instructions [here](https://github.com/facebookresearch/xformers/tree/main).


### Other Dependencies

Please install the following dependencies:

```
pip install -r requirements.txt
```

---

## Model Weights

### Pretrained Weights

You can download the pretrained weights `dinov2_vitb14_pretrain.pth` from [DINOv2](https://github.com/facebookresearch/dinov2) or [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth).

```sh
python convert_pt_weights.py 
```

For training, put the converted weights in the `model_weights` folder.

### Fine-tuned Weights

| Method | Dataset    | Weights | Configs |
| --- | --- | --- | --- |
| DAR | DUTS    | [dinov2_b_dar_duts.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_duts.pth) | [config](./configs/dinov2/config_dinov2_b_dar_duts.py) |
| DAR* | DUTS    | [dinov2_b_dar_distill_duts.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_distill_duts.pth) |  |
| DAR | CUHK    | [dinov2_b_dar_defocus.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_defocus.pth) | [config](./configs/dinov2/config_dinov2_b_dar_defocus.py) |
| DAR* | CUHK    | [dinov2_b_dar_distill_defocus.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_distill_defocus.pth) |  |
| DAR | COD10K, CAMO   | [dinov2_b_dar_cod.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_cod.pth) | [config](./configs/dinov2/config_dinov2_b_dar_cod.py) |
| DAR* | COD10K, CAMO    | [dinov2_b_dar_distill_cod.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_distill_cod.pth) |  |
| DAR | Kvasir    | [dinov2_b_dar_polyp.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_polyp.pth) | [config](./configs/dinov2/config_dinov2_b_dar_polyp.py) |
| DAR* | Kvasir    | [dinov2_b_dar_distill_polyp.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_distill_polyp.pth) |  |
| DAR | ISIC2017    | [dinov2_b_dar_skin.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_skin.pth) | [config](./configs/dinov2/config_dinov2_b_dar_skin.py) |
| DAR* | ISIC2017    | [dinov2_b_dar_distill_skin.pth](https://github.com/AVC2-UESTC/DAR-TR-PEFT/releases/download/weights/dinov2_b_dar_distill_skin.pth) |  |


For testing, put the pretrained weights and fine-tuned weights in the `model_weights` folder.


---

## Dataset

The following datasets are used in this paper:
- [DUTS](https://saliencydetection.net/duts/#orgf319326)
- [CUHK](http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/)
- [COD10K](https://github.com/DengPingFan/SINet/)
- [CAMO](https://drive.google.com/drive/folders/1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)
- [Kvasir](https://github.com/DebeshJha/2020-MediaEval-Medico-polyp-segmentation/tree/master)
- [ISIC2017](https://challenge.isic-archive.com/data/#2017)

---

## Quick Start

### Environment Setup

Make sure cuda 11.8 is installed in your virtual environment. Linux is recommmended.

Install pytorch

```sh
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

Install xformers

```sh
pip install xformers==0.0.22 --index-url https://download.pytorch.org/whl/cu118

# test installation (optional)
python -m xformers.info
```

Install mmcv

```sh
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
```

Other dependencies

```sh
pip install -r requirements.txt
```

### Prepare Dataset

We follow the [ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch) dataset format. Organize your dataset files as follows:

```
./datasets/dataset_name/

├── images/
│   ├── training/       # Put training images here
│   └── validation/     # Put validation images here
└── annotations/
    ├── training/       # Put training segmentation maps here 
    └── validation/     # Put validation segmentation maps here 
```

### Test

Put the model weights into the `model_weights` folder, and run the following command to test the model. 

```sh
python test.py --config config/path
# or
sh test.sh # for linux
# or
test.bat # for windows
# remember to modify the path in test.sh or test.bat
```

### Train

Put the pre-trained weights into the `model_weights` folder, and run the following command to train the model. 

```sh
python train.py --config config/path
# or
sh train.sh # for linux
# or
train.bat # for windows
# remember to modify the path in test.sh or test.bat
```


### Debug

If you want to debug the code, ckeck `train_debug.py` and `test_debug.py`.





---

## Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@InProceedings{Lei_2025_CVPR,
    author    = {Lei, Cheng and Li, Ao and Yao, Hu and Zhu, Ce and Zhang, Le},
    title     = {Rethinking Token Reduction with Parameter-Efficient Fine-Tuning in ViT for Pixel-Level Tasks},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {14954-14964}
}
```


---

## Acknowledgement

This project is based on [MMCV](https://github.com/open-mmlab/mmcv), [timm](https://github.com/huggingface/pytorch-image-models), [DINOv2](https://github.com/facebookresearch/dinov2), [MAM](https://github.com/jxhe/unify-parameter-efficient-tuning), and [DyT](https://github.com/NUS-HPC-AI-Lab/Dynamic-Tuning). We thank the authors for their valuable contributions.
