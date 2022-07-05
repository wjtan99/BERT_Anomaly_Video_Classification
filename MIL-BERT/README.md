# Power of Video Classification using BERT in Video Anomaly Detection 

## Datasets

Download UCF-Crime and ShanghaiTech data at following https://github.com/junha-kim/Learning-to-Adapt-to-Unseen-Abnormal-Activities 

## Checkpoints 
Download checkpoints at https://drive.google.com/drive/folders/1ABD8JZ__hX1Ab9W4L1yKihwykTe9dXbK?usp=sharing 


## Train 
python main.py 


## Test 
python main.py --test --pretrained=checkpoint --dataset UCF-Crime 


## To reproduce results on RGB+Flow in the paper 

CUDA_VISIBLE_DEVICES=0 python main.py --train_mode=2 --dataset=UCF-Crime --test --pretrain=ckpt/UCF-Crime/UCF-Crime-RGB+Flow-trainmode-2-divide32-True-L2Norm-1-multiCrop-False-epoch-9-auc-0.8671106209312655-bert.pkl

Ouput: 

args =  Namespace(L2Norm=0, batch_size=30, dataset='UCF-Crime', divideTo32=False, epochs=75, lr=0.0001, modality='RGB+Flow', multiCrop=False, pretrained='ckpt/UCF-Crime/UCF-Crime-RGB+Flow-trainmode-2-divide32-True-L2Norm-1-multiCrop-False-epoch-9-auc-0.8671106209312655-bert.pkl', resume=False, test=True, train_by_step=False, train_mode=2, workers=4)
cuda
key =  UCF-Crime-RGB+Flow-trainmode-2-divide32-True-L2Norm-1-multiCrop-False
Loading model =  ckpt/UCF-Crime/UCF-Crime-RGB+Flow-trainmode-2-divide32-True-L2Norm-1-multiCrop-False-epoch-9-auc-0.8671106209312655-bert.pkl
number of parameters =  27358338
auc = 0.8671106209312655, ap = 0.3199597162947764, auc2 = 0.8234631858560157, ap2 = 0.30293617739110074
time per frame = 2.818072959694916e-06,  fps_frame =354852.416634472






