import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

crop = 5 
L2_norm = False 


def process_feat_useone(feat, length):

    #print(feat.shape) 
    divided_features = []        
    for f in feat: 
        new_f = np.zeros((length, f.shape[1])).astype(np.float32) #32x5x1024 
        r = np.linspace(0, len(f), length+1, dtype=int)
        #print(r) 
        for i in range(length):
            if r[i]!=r[i+1]:
                new_f[i,:] = f[r[i],:]
            else:
                new_f[i,:] = f[r[i],:]
        divided_features.append(new_f) 

    divided_features = np.array(divided_features, dtype=np.float32)

    return divided_features


def process_feat(feat, length):

    #print(feat.shape) 
    divided_features = []        
    for f in feat: 
        new_f = np.zeros((length, f.shape[1])).astype(np.float32) #32x1024 
        r = np.linspace(0, len(f), length+1, dtype=int)
        #print(r) 
        for i in range(length):
            if r[i]!=r[i+1]:
                new_f[i,:] = np.mean(f[r[i]:r[i+1],:], 0)
            else:
                new_f[i,:] = f[r[i],:]

        divided_features.append(new_f) 

    divided_features = np.array(divided_features, dtype=np.float32)
    
    return divided_features




class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, dataset='UCF-Crime',modality='RGB+Flow',divideTo32=False, L2Norm=0, multiCrop=True, path=None):
        super(Normal_Loader, self).__init__()
             
        self.dataset = dataset 
        self.modality = modality 
        self.divideTo32 = divideTo32 #if true, divide features to 32-segment. Only applies to if the orignal feature is not divided into 32 segments 
        self.L2Norm = L2Norm 
        self.multiCrop = multiCrop 

        if dataset =='UCF-Crime':
            if path is None: 
                self.path='./DATA/UCF-Crime/'
            else:
                self.path= path 

            train_file = 'train_normal.txt'
            test_file = 'test_normalv3.txt'
        elif dataset =='UCF-Crime-RTFM':
            if path is None: 
                self.path='./DATA/RTFM/'
            else:
                self.path= path 

            train_file = 'train_normal.txt'
            test_file = 'test_normalv3.txt'

        elif dataset=='ShanghaiTech': 

            if path is None: 
                self.path='./DATA/ShanghaiTech/'
            else:
                self.path= path 
            train_file = 'train_normal.txt'
            test_file = 'test_normal_v2.txt'

        elif dataset=='XD-Violence': 

            if path is None: 
                self.path='./DATA/XD-Violence/'
            else:
                self.path= path 
            train_file = 'rgb_normal.list'
            test_file = 'rgb_test.list'

            #print(train_file,test_file)  
           
        self.is_train = is_train

        if self.is_train == 1:
            data_list = os.path.join(self.path, train_file)
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(self.path, test_file)
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            #random.shuffle(self.data_list)
            #self.data_list = self.data_list[:-10]
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:

            if self.dataset == 'XD-Violence': 
                concat_npy_5crop = [] 
                #print(self.data_list[idx]) 
                #rgb_file  = os.path.join(self.path+'i3d-features/RGB',  self.data_list[idx][:-1]+'__{}.npy'.format(j))

                npy_file_prefix =  self.data_list[idx].strip('\n')[:-5]  
                for j in range(0,crop): 
                    rgb_file = npy_file_prefix+'{}.npy'.format(j) 
                    tmp = rgb_file.split('/')  
                    tmp[-2] = 'Flow' 
                    flow_file = '/'.join(tmp) 
                    #print(rgb_file,flow_file) 
                    if self.modality == 'RGB+Flow':
                        flow_npy = np.load(flow_file, allow_pickle=True) 
                        rgb_npy = np.load(rgb_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                            flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                        #print(concat_npy.shape) 
                        concat_npy_5crop.append(concat_npy)
                    elif self.modality == 'RGB':
                        rgb_npy = np.load(rgb_file, allow_pickle=True) 
                        if self.L2Norm ==2:      
                            rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy_5crop.append(rgb_npy)
                    else: 
                        flow_npy = np.load(flow_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy_5crop.append(flow_npy)
                
                features = np.asarray(concat_npy_5crop)
                features = process_feat(features, 32)
                #print(concat_npy_5crop.shape) 
                if self.L2Norm>0: 
                    if self.modality == 'RGB+Flow':
                        len_feature = features.shape[2]//2                   
                        features[:,:,:len_feature] = features[:,:,:len_feature]/np.linalg.norm(features[:,:,:len_feature], ord=2, axis=2, keepdims=True)  
                        features[:,:,len_feature:] = features[:,:,len_feature:]/np.linalg.norm(features[:,:,len_feature:], ord=2, axis=2, keepdims=True)   
                    else:
                        features = features/np.linalg.norm(features, ord=2, axis=2, keepdims=True)  
                 
                return features 

            elif self.dataset == 'UCF-Crime-RTFM': 
                npy_file = self.data_list[idx][:-1]
                npy_file = npy_file.split('/')[-1] 
                npy_file = os.path.join(self.path,'UCF_Train_ten_crop_i3d',npy_file[:-4]+'_i3d.npy')
                #print(npy_file) 

                features = np.load(npy_file)

                if not self.multiCrop: 
                    #take the first crop only 
                    features = features[:,0:1] 
                    features = np.transpose(features,(1,0,2)) 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  

                    features = process_feat(features, 32)
                    if self.L2Norm>0: #L2 norm divided features 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = np.squeeze(features,0) 
                else: 
                    features = np.transpose(features,(1,0,2)) #ncrops x t x 2048 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = process_feat(features, 32) #ncrops x 32 x 2048 
                    if self.L2Norm>0: #L2 norm divided features 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  

                #print(features.shape) 
                return features 
               
 
            else:

                if self.modality == 'RGB+Flow':
                    rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                    flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                    features = np.concatenate([rgb_npy, flow_npy], axis=1)
                elif self.modality == 'RGB':
                    features = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                else:     
                    features = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))

                return features 
        else:
            if self.dataset == 'XD-Violence': 

                concat_npy_5crop = [] 
                #print(self.data_list[idx]) 

                #rgb_file  = os.path.join(self.path+'i3d-features/RGB',  self.data_list[idx][:-1]+'__{}.npy'.format(j))

                npy_file_prefix =  self.data_list[idx].strip('\n')[:-5]
                name   = npy_file_prefix.split('/')[-1]
                gts = [] 

                features = [] 
                for j in range(0,crop): 

                    rgb_file = npy_file_prefix+'{}.npy'.format(j) 
                    tmp = rgb_file.split('/')  
                    tmp[-2] = 'FlowTest' 
                    flow_file = '/'.join(tmp) 
                    #print(rgb_file,flow_file) 
                    if self.modality == 'RGB+Flow':
                        flow_npy = np.load(flow_file, allow_pickle=True) 
                        rgb_npy = np.load(rgb_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                            flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                        #print(concat_npy.shape) 
                        concat_npy_5crop.append(concat_npy)
                    elif self.modality == 'RGB':
                        rgb_npy = np.load(rgb_file, allow_pickle=True) 
                        if self.L2Norm ==2:      
                            rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy_5crop.append(rgb_npy)
                    else: 
                        flow_npy = np.load(flow_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy_5crop.append(flow_npy)
                
                features = np.asarray(concat_npy_5crop)
                frames = features.shape[-2]*16 
                if self.divideTo32: 
                    #print(concat_npy_5crop.shape) 
                    features = process_feat(features, 32)
                    #print(concat_npy_5crop.shape) 
                if self.L2Norm>0: 
                    if self.modality == 'RGB+Flow':
                        len_feature = features.shape[2]//2                   
                        features[:,:,:len_feature] = features[:,:,:len_feature]/np.linalg.norm(features[:,:,:len_feature], ord=2, axis=2, keepdims=True)  
                        features[:,:,len_feature:] = features[:,:,len_feature:]/np.linalg.norm(features[:,:,len_feature:], ord=2, axis=2, keepdims=True)   
                    else:
                        features = features/np.linalg.norm(features, ord=2, axis=2, keepdims=True)  

                return features, gts, frames, name  

            elif self.dataset == 'UCF-Crime-RTFM': 

                name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])

                npy_file = name
                npy_file = npy_file.split('/')[-1] 
                npy_file = os.path.join(self.path,'UCF_Test_ten_crop_i3d',npy_file[:-4]+'_i3d.npy')
                #print(npy_file) 

                features = np.load(npy_file)

                if not self.multiCrop: 
                    #take the first crop only 
                    features = features[:,0:1] 
                    features = np.transpose(features,(1,0,2)) 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    if self.divideTo32: 
                        features = process_feat(features, 32)
                        if self.L2Norm>0: #L2 norm divided features 
                            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = np.squeeze(features,0) 
                else: 
                    features = np.transpose(features,(1,0,2)) #ncrops x t x 2048 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    if self.divideTo32: 
                        features = process_feat(features, 32) #ncrops x 32 x 2048 
                        if self.L2Norm>0: #L2 norm divided features 
                            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                #print(features.shape) 
                return features, gts, frames, name

            else:
                name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
                #print(name,frames,gts) 

                if self.modality == 'RGB+Flow':
                    rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name +'.npy'))
                    flow_npy = np.load(os.path.join(self.path+'all_flows', name +'.npy'))
                    features = np.concatenate([rgb_npy, flow_npy], axis=1)
                elif self.modality == 'RGB':
                    features = np.load(os.path.join(self.path+'all_rgbs', name +'.npy'))
                else:     
                    features = np.load(os.path.join(self.path+'all_flows', name +'.npy'))

                return features, gts, frames, name

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1, dataset='UCF-Crime',modality='RGB+Flow',divideTo32=False, L2Norm=0, multiCrop=True, path=None):
        super(Anomaly_Loader, self).__init__()

        self.dataset = dataset 
        self.modality = modality 
        self.divideTo32 = divideTo32 #if true, divide features to 32-segment. Only applies to if the orignal feature is not divided into 32 segments 
        self.L2Norm = L2Norm 
        self.multiCrop = multiCrop 

        if dataset =='UCF-Crime':
            if path is None: 
                self.path='./DATA/UCF-Crime/'
            else:
                self.path= path 

            train_file = 'train_anomaly.txt'
            test_file = 'test_anomalyv3.txt'
        elif dataset =='UCF-Crime-RTFM':
            if path is None: 
                self.path='./DATA/RTFM/'
            else:
                self.path= path 

            train_file = 'train_anomaly.txt'
            test_file = 'test_anomalyv3.txt'
        elif dataset=='ShanghaiTech': 

            if path is None: 
                self.path='./DATA/ShanghaiTech/'
            else:
                self.path= path 

            train_file = 'train_anomaly.txt'
            test_file = 'test_anomaly.txt'
        elif dataset=='XD-Violence': 

            if path is None: 
                self.path='./DATA/XD-Violence/'
            else:
                self.path= path 
            train_file = 'rgb_abnormal.list'
            test_file = 'rgb_test.list'


        self.is_train = is_train

        if self.is_train == 1:
            data_list = os.path.join(self.path, train_file)            
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(self.path, test_file)
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        #print(data_list) 
        #print(self.data_list) 
 
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            if self.dataset == 'XD-Violence': 
                concat_npy_5crop = [] 
                #print(self.data_list[idx])
                #rgb_file  = os.path.join(self.path+'i3d-features/RGB',  self.data_list[idx][:-1]+'__{}.npy'.format(j))
                npy_file_prefix =  self.data_list[idx].strip('\n')[:-5]  

                for j in range(0,crop): 
                    rgb_file = npy_file_prefix+'{}.npy'.format(j) 
                    tmp = rgb_file.split('/')  
                    tmp[-2] = 'Flow' 
                    flow_file = '/'.join(tmp) 

                    if self.modality == 'RGB+Flow':
                        flow_npy = np.load(flow_file, allow_pickle=True) 
                        rgb_npy = np.load(rgb_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                            flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                        #print(concat_npy.shape) 
                        concat_npy_5crop.append(concat_npy)
                    elif self.modality == 'RGB':
                        rgb_npy = np.load(rgb_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy_5crop.append(rgb_npy)
                    else: 
                        flow_npy = np.load(flow_file, allow_pickle=True) 
                        if self.L2Norm == 2:      
                            flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                        concat_npy_5crop.append(flow_npy)
                
                features = np.asarray(concat_npy_5crop)
                features = process_feat(features, 32)
                #print(concat_npy_5crop.shape) 
                if self.L2Norm>0: 
                    if self.modality == 'RGB+Flow':
                        len_feature = features.shape[2]//2                   
                        features[:,:,:len_feature] = features[:,:,:len_feature]/np.linalg.norm(features[:,:,:len_feature], ord=2, axis=2, keepdims=True)  
                        features[:,:,len_feature:] = features[:,:,len_feature:]/np.linalg.norm(features[:,:,len_feature:], ord=2, axis=2, keepdims=True)   
                    else:
                        features = features/np.linalg.norm(features, ord=2, axis=2, keepdims=True)                   
                 
                return features 
            elif self.dataset == 'UCF-Crime-RTFM': 
                npy_file = self.data_list[idx][:-1]
                npy_file = npy_file.split('/')[-1] 
                npy_file = os.path.join(self.path,'UCF_Train_ten_crop_i3d',npy_file[:-4]+'_i3d.npy')
                #print(npy_file) 

                features = np.load(npy_file)
                if not self.multiCrop: 
                    #take the first crop only 
                    features = features[:,0:1] 
                    features = np.transpose(features,(1,0,2)) 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = process_feat(features, 32)
                    if self.L2Norm>0: #L2 norm divided features 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = np.squeeze(features,0) 
                else: 
                    features = np.transpose(features,(1,0,2)) #ncrops x t x 2048 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = process_feat(features, 32) #ncrops x 32 x 2048 
                    if self.L2Norm>0: #L2 norm divided features 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                #print(features.shape) 
                return features 
            else:
                if self.modality == 'RGB+Flow':
                    rgb_npy = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                    flow_npy = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                    features = np.concatenate([rgb_npy, flow_npy], axis=1)
                elif self.modality == 'RGB':
                    features = np.load(os.path.join(self.path+'all_rgbs', self.data_list[idx][:-1]+'.npy'))
                else:     
                    features = np.load(os.path.join(self.path+'all_flows', self.data_list[idx][:-1]+'.npy'))
                #print(features.shape) 
                return features
        else:

            if self.dataset == 'XD-Violence': 
                concat_npy_5crop = [] 

                #print(self.data_list[idx]) 

                name   = self.data_list[idx].split('|')[0] 
                gts    = self.data_list[idx].split('|')[2][1:-2].split(',')
                gts = [int(i) for i in gts]

                frames = int(self.data_list[idx].split('|')[1])
                #print(frames, gts)
               
                for j in range(crop):
                    rgb_file  = os.path.join(self.path+'i3d-features/RGBTest',  name+'__{}.npy'.format(j))
                    flow_file = os.path.join(self.path+'i3d-features/FlowTest', name+'__{}.npy'.format(j))
                    rgb_npy  = np.load(rgb_file) 
                    flow_npy = np.load(flow_file) 
                    rgb_npy = rgb_npy/np.linalg.norm(rgb_npy, ord=2, axis=-1, keepdims=True)  
                    flow_npy = flow_npy/np.linalg.norm(flow_npy, ord=2, axis=-1, keepdims=True)  
                    concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)
                    #print(concat_npy.shape) 
                    concat_npy_5crop.append(concat_npy) 
                concat_npy_5crop = np.asarray(concat_npy_5crop)
                
                #Changed to be processed outside 
                #print(concat_npy_5crop.shape) 
                concat_npy_5crop = process_feat(concat_npy_5crop, 32)
                #print(concat_npy_5crop.shape) 
                 
                return concat_npy_5crop, gts, frames, name  

            elif self.dataset == 'UCF-Crime-RTFM': 

                name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
                gts = [int(i) for i in gts]

                npy_file = name
                npy_file = npy_file.split('/')[-1] 
                npy_file = os.path.join(self.path,'UCF_Test_ten_crop_i3d',npy_file[:-4]+'_i3d.npy')
                #print(npy_file) 

                features = np.load(npy_file)
                if not self.multiCrop: 
                    #take the first crop only 
                    features = features[:,0:1] 
                    features = np.transpose(features,(1,0,2)) 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    if self.divideTo32: 
                        features = process_feat(features, 32)
                        if self.L2Norm>0: #L2 norm divided features 
                            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    features = np.squeeze(features,0) 
                else: 
                    features = np.transpose(features,(1,0,2)) #ncrops x t x 2048 
                    if self.L2Norm==2: #L2 norm every feature 
                        features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                    if self.divideTo32: 
                        features = process_feat(features, 32) #ncrops x 32 x 2048 
                        if self.L2Norm>0: #L2 norm divided features 
                            features = features/np.linalg.norm(features, ord=2, axis=-1, keepdims=True)  
                #print(features.shape) 
                return features, gts, frames,name

            else:
                name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
                gts = [int(i) for i in gts]

   
                if self.modality == 'RGB+Flow':
                    rgb_npy = np.load(os.path.join(self.path+'all_rgbs', name +'.npy'))
                    flow_npy = np.load(os.path.join(self.path+'all_flows', name +'.npy'))
                    features = np.concatenate([rgb_npy, flow_npy], axis=1)
                elif self.modality == 'RGB':
                    features = np.load(os.path.join(self.path+'all_rgbs', name +'.npy'))
                else:     
                    features = np.load(os.path.join(self.path+'all_flows', name +'.npy'))

                return features, gts, frames,name

if __name__ == '__main__':

    loader = Normal_Loader(is_train=1, dataset='UCF-Crime',modality='RGB',divideTo32=False, L2Norm=0)
    #print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    #print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

    loader = Normal_Loader(is_train=1, dataset='UCF-Crime-RTFM',divideTo32=False, L2Norm=0)
    print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    #print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  
    loader = Normal_Loader(is_train=1, dataset='UCF-Crime-RTFM',divideTo32=True, L2Norm=0)
    print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

    loader = Normal_Loader(is_train=1, dataset='UCF-Crime-RTFM',divideTo32=True, L2Norm=1)
    print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

    loader = Normal_Loader(is_train=1, dataset='UCF-Crime',modality='RGB+Flow',divideTo32=False, L2Norm=0)
    print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

    exit(0) 

    loader = Normal_Loader(is_train=1, dataset='XD-Violence')
    #print(len(loader))
    print(loader[0].shape,loader[0][0].shape)

    loader = Normal_Loader(is_train=1, dataset='XD-Violence',divideTo32=True, L2Norm=0)
    #print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

    loader = Normal_Loader(is_train=1, dataset='XD-Violence',divideTo32=True, L2Norm=0)
    #print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  


    loader = Normal_Loader(is_train=1, dataset='UCF-Crime',divideTo32=False, L2Norm=0)
    #print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

    loader = Normal_Loader(is_train=1, dataset='UCF-Crime',modality='RGB',divideTo32=False, L2Norm=0)
    #print(len(loader))
    print(loader[0].shape,loader[0][0].shape)
    print(np.linalg.norm(loader[0], ord=2, axis=-1, keepdims=True))  

