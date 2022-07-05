import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as torch_init

from BERT.bert import  BERT5

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        '''
        if m.bias is not None:
            m.bias.data.fill_(0)
        ''' 
class Learner(nn.Module):
    def __init__(self,  modality='RGB+Flow', feature_dim=1024, BERT_Enable=True):
        super(Learner, self).__init__()

        self.bert_input_dim = feature_dim
        self.modality = modality 
        self.BERT_Enable = BERT_Enable 

        if modality =='RGB+Flow':
            self.cnn_input_dim = feature_dim*2 
        else: 
            self.cnn_input_dim = feature_dim 
       
        self.fc1 = nn.Linear(self.cnn_input_dim, 512)
        self.fc2 = nn.Linear(512,32)
        self.fc3 = nn.Linear(32, 1)

        self.dp = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if self.BERT_Enable: 
            self.fc1_2 = nn.Linear(self.cnn_input_dim, 512)
            self.fc2_2 = nn.Linear(512,32)
            self.fc3_2 = nn.Linear(32, 1)
            self.bert = BERT5(self.bert_input_dim,32,hidden=self.bert_input_dim, n_layers=2, attn_heads=8)

        self.apply(weight_init)

    def forward(self, x):
       
        debug = False 
        if debug: 
            print("input.shape = ", x.shape) 

        num_segs = x.shape[-2] 
        
        if self.modality == 'RGB' or self.modality == 'Flow':            
            x0 = x

            if self.BERT_Enable and num_segs==32:
                output1, mask1 = self.bert(x)
                cls  = output1[:,0,:]
                yseq = output1[:,1:,:]

        else: 
            x0 = x  
            #print("input.shape = ", x.shape,x0.shape,self.input_dim) 

            if self.BERT_Enable and num_segs==32:

                x1 = x[:,:,:self.bert_input_dim] 
                x2 = x[:,:,self.bert_input_dim:] 
 
                #print("x1/2.shape = ", x1.shape,x2.shape) 
      
                output1, mask1 = self.bert(x1)
                cls1  = output1[:,0,:]
                yseq1 = output1[:,1:,:]

                output2, mask2 = self.bert(x2)
                cls2  = output2[:,0,:]
                yseq2 = output2[:,1:,:]

                cls = torch.cat((cls1,cls2),-1) 
                yseq = torch.cat((yseq1,yseq2),-1) 
       
        if self.BERT_Enable and num_segs==32 and True: 
            norm = yseq.norm(p=2, dim = -1, keepdim=True)
            yseq = yseq.div(norm)
            norm = cls.norm(p=2, dim = -1, keepdim=True)
            cls = cls.div(norm)         

            if debug: 
                print("yseq.shape = ", yseq.shape) 
                print("cls.shape = ", cls.shape) 


        x = x0 # x0 #yseq is a lot worser 
        x = self.relu(self.fc1(x))
        x = self.dp(x) 
        #WT 06/13/22 the performance is worse when relu is used on fc2 
        x = self.fc2(x)
        x = self.dp(x) 
        out1 = self.sigmoid(self.fc3(x))    

        out2 = None 
        if self.BERT_Enable  and num_segs==32:
            x = cls    
            x = self.relu(self.fc1_2(x))
            x = self.dp(x) 
            #WT 06/13/22 the performance is worse when relu is used on fc2 
            x = self.fc2_2(x)
            x = self.dp(x) 
            out2  = self.sigmoid(self.fc3_2(x))
            if debug:
                print("output1 x.shape = ", out1.shape) 
                print("output2 x.shape = ", out2.shape) 

        return out1, out2 


class Learner_multicrop(nn.Module):
    def __init__(self,  modality='RGB+Flow', feature_dim=1024, BERT_Enable=True):
        super(Learner_multicrop, self).__init__()
        
        self.modality = modality 

        if modality =='RGB+Flow':
            self.cnn_input_dim = feature_dim*2 
        else: 
            self.cnn_input_dim = feature_dim

        self.BERT_Enable = BERT_Enable 
        self.bert_input_dim = feature_dim
       
        self.fc1 = nn.Linear(self.cnn_input_dim, 512)
        self.fc2 = nn.Linear(512,32)
        self.fc3 = nn.Linear(32, 1)

        self.dp = nn.Dropout(0.6)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if BERT_Enable: 
            self.bert = BERT5(self.bert_input_dim,32,hidden=self.bert_input_dim, n_layers=2, attn_heads=8)
            self.fc1_2 = nn.Linear(self.cnn_input_dim, 512)
            self.fc2_2 = nn.Linear(512,32)
            self.fc3_2 = nn.Linear(32, 1)

        self.apply(weight_init)


    def forward(self, x):
       
        debug = False 
        if debug: 
            print("input.shape = ", x.shape) 

        num_segs = x.shape[-2] 


        if self.modality == 'RGB' or self.modality == 'Flow': 

            x0 = x 

            bs, ncrops, t, f = x.size()

            x = x.view(-1, t, f)
            if debug: print("out1.shape = ", x.shape) 

            x = self.relu(self.fc1(x))
            x = self.dp(x) 
            #WT 06/13/22 the performance is worse when relu is used on fc2 
            x = (self.fc2(x))
            x = self.dp(x) 
            x = self.sigmoid(self.fc3(x))    

            if debug: print("FC score.shape = ", x.shape) 
            x = x.view(bs, ncrops, -1)
            if debug: print("FC score view.shape = ", x.shape) 
            x = x.mean(1) 
            if debug: print("FC score crop mean.shape = ", x.shape) 
            out1 = x.unsqueeze(dim=2)
            if debug: print("FC score final shape = ", x.shape) 


            out2 = None 
            if num_segs==32 and self.BERT_Enable: 
   
                bs, ncrops, t, f = x0.size()
                x = x0.view(-1, t, f)
 
                #print("x.shape = ", x.shape)      

                output1, mask1 = self.bert(x)
                cls  = output1[:,0,:]
                yseq = output1[:,1:,:]

                norm = cls.norm(p=2, dim = -1, keepdim=True)
                cls = cls.div(norm)     

                #print("cls.shape = ", cls.shape)      

                x = self.relu(self.fc1_2(cls))
                x = self.dp(x) 
                #WT 06/13/22 the performance is worse when relu is used on fc2 
                x = self.fc2_2(x)
                x = self.dp(x) 
                x = self.sigmoid(self.fc3_2(x))    

                if debug: print("FC score.shape = ", x.shape) 
                x = x.view(bs, ncrops, -1)
                if debug: print("FC score view.shape = ", x.shape) 
                out2 = x.mean(1) 
                if debug: print("FC score crop mean.shape = ", x.shape) 
            if debug: print(out1.shape,out2.shape) 

        else: #RGB+Flow 

            x0  = x  
            if debug: print("input.shape = ", x.shape) 

            bs, ncrops, t, f = x.size()

            x = x.view(-1, t, f)
            if debug: print("out1.shape = ", x.shape) 

            x = self.relu(self.fc1(x))
            x = self.dp(x) 
            #WT 06/13/22 the performance is worse when relu is used on fc2 
            x = self.fc2(x)
            x = self.dp(x) 
            x = self.sigmoid(self.fc3(x))    

            if debug: print("FC score.shape = ", x.shape) 
            x = x.view(bs, ncrops, -1)
            if debug: print("FC score view.shape = ", x.shape) 
            x = x.mean(1) 
            if debug: print("FC score crop mean.shape = ", x.shape) 
            out1 = x.unsqueeze(dim=2)
            if debug: print("FC score final shape = ", x.shape) 


            out2 = None 
            if num_segs==32 and self.BERT_Enable: 
   
                bs, ncrops, t, f = x0.size()
                x = x0.view(-1, t, f)

                x1 = x[:,:,:f//2] 
                x2 = x[:,:,f//2:]  

                if debug: print("x1/2.shape = ", x1.shape,x2.shape)      

                output1, mask1 = self.bert(x1)
                cls1  = output1[:,0,:]
                yseq1 = output1[:,1:,:]

                output2, mask2 = self.bert(x2)
                cls2  = output2[:,0,:]
                yseq2 = output2[:,1:,:]

                cls = torch.cat((cls1,cls2),-1) 
                yseq = torch.cat((yseq1,yseq2),-1) 

                norm = cls.norm(p=2, dim = -1, keepdim=True)
                cls = cls.div(norm)     

                if debug: print("cls.shape = ", cls.shape)      

                x = self.relu(self.fc1_2(cls))
                x = self.dp(x) 
                #WT 06/13/22 the performance is worse when relu is used on fc2 
                x = self.fc2_2(x)
                x = self.dp(x) 
                x = self.sigmoid(self.fc3_2(x))    

                if debug: print("FC score.shape = ", x.shape) 
                x = x.view(bs, ncrops, -1)
                if debug: print("FC score view.shape = ", x.shape) 
                out2 = x.mean(1) 
                if debug: print("FC score crop mean.shape = ", x.shape) 

            if debug: print(out1.shape,out2.shape) 


        return out1,out2 


