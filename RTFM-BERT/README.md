# RTFM-BERT 

## Datasets

Follow https://github.com/tianyu0207/RTFM to download UCF-Crime and ShanghaiTech feature sets
 
Follow https://github.com/Roc-Ng/XDVioDet to download XD-Violence feature sets 

## Train 
python main.py

## Checkpoints 
Download checkpoints at https://drive.google.com/drive/folders/1ABD8JZ__hX1Ab9W4L1yKihwykTe9dXbK?usp=sharing 

## Reproduce results on XD-Violence 

python main.py --dataset=XDViolence --test  --modality=RGB  --feature-size=1024 --pretrained=ckpt/rtfm-bert-i3d-XDViolence-beta-0.5-step-10270-AP-0.8210568287223814.pkl

Output: 
time_per_frame =  5.525224505887171e-06
auc = 0.910260592674212, ap = 0.7776958804308781, auc2 = 0.9320619330643101, ap2 = 0.8210568287223814 







