# HNeRV: A Hybrid Neural Representation for Videos  (Under Review)
### [Project Page](https://haochen-rye.github.io/HNeRV) | [UVG Data](http://ultravideo.fi/#testsequences) 


[Hao Chen](https://haochen-rye.github.io),
Matthew Gwilliam,
Ser-Nam Lim,
[Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)<br>
This is the official implementation of the paper "HNeRV: A Hybrid Neural Representation for Videos".

## TODO 
- [ &check; ] Video inpainting
- [ &check; ] Fast loading from video checkpoints
- [ ] Upload results and checkpoints for UVG

## Method overview
<!-- HNeRV architecture             |  Video Regression
:-------------------------:|:-------------------------: -->
![](https://i.imgur.com/SdRcEiY.jpg) 

 ![](https://i.imgur.com/CAppWSM.jpg)

## Get started
We run with Python 3.8, you can set up a conda environment with all dependencies like so:
```
pip install -r requirements.txt 
```

## High-Level structure
The code is organized as follows:
* [train_nerv_all.py](./train_nerv_all.py) includes a generic traiing routine.
* [model_all.py](./model_all.py) contains the dataloader and neural network architecure 
* [data/](./data) directory video/imae dataset, we provide big buck bunny here
* log files (tensorboard, txt, state_dict etc.) will be saved in output directory (specified by ```--outf```)

## Reproducing experiments

### Training HNeRV
HNeRV of 1.5M is specified with ```'--modelsize 1.5'```, and we balance parameters with ```'-ks 0_1_5 --reduce 1.2' ```
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

### NeRV baseline
NeRV baseline is specified with ```'--embed pe_1.25_80 --fc_hw 8_16'```, with imbalanced parameters ```'--ks 0_3_3 --reduce 2' ```
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
   --resize_list -1 --loss L2   --embed pe_1.25_80 --fc_hw 8_16 \
    --dec_strds 5 4 2 2 --ks 0_3_3 --reduce 2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

### Evaluation & dump images and videos
To evaluate pre-trained model, use ```'--eval_only --weight [CKT_PATH]'``` to evaluate and specify model path. \
For model and embedding quantization, use ```'--quant_model_bit 8 --quant_embed_bit 6'```.\
To dump images or videos, use  ```'--dump_images --dump_videos'```.
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2  \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001 \
   --eval_only --weight checkpoints/hnerv-1.5m-e300.pth \
   --quant_model_bit 8 --quant_embed_bit 6 \
    --dump_images --dump_videos
```

### Video inpainting
We can specified inpainting task with ```'--vid bunny_inpaint_50'``` where '50' is the mask size.
```
python train_nerv_all.py  --outf 1120  --data_path data/bunny --vid bunny_inpaint_50   \
   --conv_type convnext pshuffel --act gelu --norm none  --crop_list 640_1280  \
    --resize_list -1 --loss L2  --enc_strds 5 4 4 2 2 --enc_dim 64_16 \
    --dec_strds 5 4 4 2 2 --ks 0_1_5 --reduce 1.2   \
    --modelsize 1.5  -e 300 --eval_freq 30  --lower_width 12 -b 2 --lr 0.001
```

## Citation
If you find our work useful in your research, please cite:
```
@InProceedings{chen2022hnerv,
      title={{HN}e{RV}: Neural Representations for Videos}, 
      author={Hao Chen and Matthew Gwilliam and Ser-Nam Lim and Abhinav Shrivastava},
      year={2022},
}
```

## Contact
If you have any questions, please feel free to email the authors.