# GazeHTA
## Getting Started 

1. Create environment

    ```
    conda create python=3.8.5 -n gazehta
    ```

2. Install the required packages.

    ```
    conda activate gazehta
    conda install pytorch=1.13.1=py3.8_cuda11.7_cudnn8.5.0_0 torchvision=0.14.1=py38_cu117 torchaudio=0.13.1=py38_cu117 cudatoolkit=11.3.1=h2bc3f7f_2 -c pytorch -c nvidia
    
    pip install pandas tensorboardX timm tokenizers==0.13.2 torchmetrics==0.6.0 einops omegaconf transformers scipy scikit-image tqdm matplotlib h5py  diffusers opencv-python pytorch-lightning==1.4.2 taming-transformers-rom1504 mmengine 
    ```
    

3. Prepare datasets

    * GazeFollow: we use the extended GazeFollow annotation prepared by Chong et al. ECCV 2018, which makes an additional annotation to the original GazeFollow dataset regarding whether gaze targets are within or outside the frame. You can download the extended dataset from [here](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?e=1&dl=0).
    
    * VideoAttentionTarget: download [here](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?e=1&dl=0)
    
    * Pseudo labels from [YOLO-v5](https://github.com/xinsuinizhuan/yolov5-head-detector). The annotations can be found in the folder 'datasets'.
    
    * The validation list for each dataset can also be found in the folder 'datasets'.

    * Your dataset directory should look like:
    
    ```
    home_dir/
    ├──train_data/
    │  ├── gazefollow/
    │  │   ├── test2/
    │  │   ├── train/
    │  │   ├── train_annotations_release.txt
    │  │   ├── test_annotations_release.txt
    │  │   ├── train_heads_yolov5.csv
    │  │   ├── test_heads_yolov5.csv
    │  │   ├── val_list_gazefollow.npy
    │  ├── videoattentiontarget/
    │  │   ├── annotations/
    │  │   ├── images/
    │  │   ├── train_heads_yolov5.csv
    │  │   ├── test_heads_yolov5.csv
    │  │   ├── val_clip_list_videoatttarget.npy
    ├──...
    ```


## Training

**[Update 11/18/2024]**
The StableDiffusion-v1.5 pre-trained model is in the checkpoints folder since the link to StableDiffusion-v1.5 is deprecated. You can find some information [here](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).

* Download the checkpoint of [stable-diffusion](https://github.com/runwayml/stable-diffusion) (v1-5) and put it in the checkpoints folder. Please also follow the instructions in stable-diffusion to install the required packages.


* Modify your settings such as datasets, working directories, and output paths, in run_train.py first. 

    ```
    python3.8 run_train.py
    ```

## Evaluation

* The pre-trained models for GazeFollow and VideoAttentionTarget can be found [here](https://drive.google.com/drive/folders/1s80QqLELewJy0qZLfjHGWTqo5PGlHYeM?usp=sharing). Place the pre-trained models in the 'pretrained_models' folder.

* Change the setting in run_train.py to:

```
eval_only = 1 # whether to only run evaluation or not
use_pretrained = 1 # whether to initialize the model with a pretrained model
checkpoint_filename =  f'./pretrained_models/model_videoAttTarget_GazeHTA_epoch_9.pth' # path to the pre-trained model
```
* Then, run the script:

```
python3.8 run_train.py
```

## Acknowledgements
This implementation is based on [Object-aware Gaze Target Detection](https://github.com/francescotonini/object-aware-gaze-target-detection).
