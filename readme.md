<h1 align="center">[CVPR 2025] Rethinking Token Reduction with Parameter-Efficient Fine-Tuning in ViT for Pixel-Level Tasks</h1>

<div align="center">
  <hr>
  Cheng Lei &nbsp;
  Ao Li &nbsp;
  Hu Yao &nbsp;
  Ce Zhu &nbsp;
  Le Zhang &nbsp;
  <br>
    University of Electronic Science and Technology of China &nbsp;

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


## Installation

It is reconmended to install the newest version of Pytorch and MMCV.

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

### Best Practice

Make sure cuda 11.8 is installed in your vitual environment. Linux is recommmended.

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

Others

```sh
pip install -r requirements.txt
```



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
