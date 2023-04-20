# A Progressive Framework for Rotary Motion Deblurring


## Overview
The rotary motion deblurring is an inevitable procedure when the imaging seeker is mounted in the rotating missiles. Traditional rotary motion deblurring methods suffer from ringing artifacts and noise, especially for large blur extents. To solve the above problems, we propose a progressive rotary motion deblurring framework consisting of a coarse deblurring stage and a refinement stage. To establish a standard evaluation benchmark, a real-world rotary motion blur dataset is proposed and released, which includes rotary blurred images and corresponding ground truth images with different blur angles. Experimental results demonstrate that the proposed method outperforms the state-of-the-art models on synthetic and real-world rotary motion blur datasets.

## Datasets
### Synthetic rotary motion blur datasets
We use BSDS500 ([Baidu Netdisk](https://pan.baidu.com/s/1MGoNMQJd2auIfLObrT9YTQ?pwd=rmbd) or [Google drive](https://drive.google.com/file/d/1fZIkxHV_DB0ZZ-D3u7RtNzahkPwD76Ua/view?usp=sharing))  datasets as the ground truth images to generate synthetic rotary motion burring datasets (S-RMBD). Assume BSDS500 path is '/syn_sharp/', and run the following code to obtain datasets (results of the coarse deblurring stage):
```
python create_datasets.py --dataset syn --data_root /syn_sharp/ --save_root /path/to/save/syn/
```
### Real-world rotary motion blur datasets with ground truth
Our real-world rotary motion blur datasets (R-RMBD) can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1XSLMNu6mA_UNeEb6manJpg?pwd=rmbd) or [Google drive](https://drive.google.com/file/d/1W_W2z86UBhrS7-a-GWnSMpB6jF5StRgF/view?usp=sharing). Assume R-RMBD path is '/real_sharp/', and run the following code to obtain datasets (results of the coarse deblurring stage):
```
python create_datasets.py --dataset real --data_root /real_sharp/ --save_root /path/to/save/real/
```
### Real-world rotary motion blur image without ground truth
Our real-world rotary motion blur image and its reference ground truth can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/1ban3OXMgmRQ5WTvkJjUbnw?pwd=rmbd) or [Google drive](https://drive.google.com/file/d/1BXQOA8fVLT9TRNWUIvjenSSsXpzwq-Jm/view?usp=sharing).

### Training
To train S-RMBD, run the following code:
```
python main.py --data_root /path/to/save/syn/ --dataset BSDS500
```
To train R-RMBD, run the following code:
```
python main.py --data_root /path/to/save/real/ --dataset real_world
```
### Evaluation
Models can be downloaded from [Baidu Netdisk](https://pan.baidu.com/s/134HOeU_6K51YUbqdLEQYDg?pwd=rmbd) or [Google drive](https://drive.google.com/file/d/1oKfhn8ZuI31Om81DEMHHJRmMy5u6HkNV/view?usp=share_link).
To evaluate a synthetic rotary motion burred image, run the following code:
```
python evl.py --dataset syn --path_model /path/to/syn.pt
```
To evaluate a real-world rotary motion burred image, run the following code:
```
python evl.py --dataset real --path_model /path/to/real.pt
```

### Acknowledgements
https://github.com/SeungjunNah/DeepDeblur-PyTorch