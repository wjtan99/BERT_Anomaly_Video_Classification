from torch.utils.data import DataLoader
from learner import Learner,Learner_multicrop
from loss import *
from dataset import *
import os

from sklearn import metrics # import auc, roc_curve, precision_recall_curve, average_precision_score 

import argparse
from tqdm import tqdm
import time 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

parser = argparse.ArgumentParser(description='MIL-BERT')

parser.add_argument('--modality', default='RGB+Flow', help='the type of the input, RGB,Flow,or RGB+Flow')
parser.add_argument('--batch-size', type=int, default=30, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--pretrained', default='model_best-joint-e2e-RGB+Flow.pkl', help='ckpt for pretrained model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--dataset', default='UCF-Crime', help='dataset to train on (default: )')
parser.add_argument('--epochs', default=75, type=int, help='number of epochs in dataloader')
parser.add_argument('--train_mode', default=2, type=int, help='train mode')

parser.add_argument('-r', '--resume', dest='resume', action='store_true',help='continue to train')
parser.add_argument('-t', '--test', dest='test', action='store_true',help='test')

parser.add_argument('--divideTo32', dest='divideTo32', action='store_true',help='divideTo32')
parser.add_argument('--L2Norm', default=0, type=int, help='L2Norm')
parser.add_argument('--multiCrop', dest='multiCrop', action='store_true',help='multiCrop') #only test on UCF-Crime-RTFM 
parser.add_argument('--train_by_step', dest='train_by_step', action='store_true',help='train by step')  



args = parser.parse_args()
print('args = ', args) 


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) 


if args.dataset=='XD-Violence':
    normal_train_dataset = Normal_Loader(is_train=1,dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm)
    normal_test_dataset = Normal_Loader(is_train=0, dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm)

    anomaly_train_dataset = Anomaly_Loader(is_train=1,dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm)

    if args.train_mode==0:
        model = Learner_multicrop(feature_dim=1024, modality=args.modality, BERT_Enable =False)
    else: 
        model = Learner_multicrop(feature_dim=1024, modality=args.modality)

elif args.dataset == 'UCF-Crime-RTFM': 

    normal_train_dataset = Normal_Loader(is_train=1,dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm, multiCrop=args.multiCrop)
    normal_test_dataset = Normal_Loader(is_train=0, dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm, multiCrop=args.multiCrop)

    anomaly_train_dataset = Anomaly_Loader(is_train=1,dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm, multiCrop=args.multiCrop)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, dataset=args.dataset,modality=args.modality,divideTo32=args.divideTo32, L2Norm=args.L2Norm, multiCrop=args.multiCrop)

    args.modality = 'RGB' 

    #test 1 crop only 
    if not args.multiCrop: 
        if args.train_mode==0:
            model = Learner(feature_dim=2048, modality=args.modality, BERT_Enable =False)
        else: 
            model = Learner(feature_dim=2048, modality=args.modality)
    else: 
        if args.train_mode==0:
            model = Learner_multicrop(feature_dim=2048, modality=args.modality, BERT_Enable =False)
        else: 
            model = Learner_multicrop(feature_dim=2048, modality=args.modality)

else:

    #the data is already divided into 32 segments, and L2 normalized. 
    args.divideTo32 = True
    args.L2Norm = 1 
    args.multiCrop = False  

    normal_train_dataset = Normal_Loader(is_train=1,dataset=args.dataset,modality=args.modality)
    normal_test_dataset = Normal_Loader(is_train=0, dataset=args.dataset,modality=args.modality)

    anomaly_train_dataset = Anomaly_Loader(is_train=1,dataset=args.dataset,modality=args.modality)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, dataset=args.dataset,modality=args.modality)

    if args.train_mode==0:
        model = Learner(feature_dim=1024, modality=args.modality, BERT_Enable =False)
    else: 
        model = Learner(feature_dim=1024, modality=args.modality)

key = '{}-{}-trainmode-{}-divide32-{}-L2Norm-{}-multiCrop-{}'.format(args.dataset,args.modality,args.train_mode,args.divideTo32,args.L2Norm,args.multiCrop)
print('key = ',key) 



normal_train_loader = DataLoader(normal_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False, num_workers=4,pin_memory=True)

anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers,pin_memory=True) 
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False,  num_workers=4,pin_memory=True)



if torch.cuda.device_count() > 1:
    print("Using multiple GPUs") 
    model=torch.nn.DataParallel(model)

model = model.to(device) 


optimizer = torch.optim.Adagrad(model.parameters(), lr= args.lr, weight_decay=0.0010000000474974513)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,200])
criterion = MIL
criterion_video = torch.nn.BCELoss()


def divide_feature(feature, length):

    debug = False 

    feat = torch.squeeze(feature,0) #remove batch, for batch=1 only 
    if debug:
        print(feat.shape) 

    #print(feat.shape) 
    divided_features = torch.zeros((feat.shape[0],32,feat.shape[-1]))   
    r = np.linspace(0, feat.shape[1], length+1, dtype=int)
    if debug: 
        print(r) 
   
    for ind in range(feat.shape[0]): 
        f = feat[ind] 
        new_f = torch.zeros((length, f.shape[-1])) #32x1024 

        for i in range(length):
            if r[i]!=r[i+1]:
                new_f[i,:] = torch.mean(f[r[i]:r[i+1],:], 0)
            else:
                new_f[i,:] = f[r[i],:]
        if debug: 
            print(new_f.shape) 

        divided_features[ind] = new_f 

    if args.L2Norm>0: 
        if args.modality=='RGB+Flow':
            len_feature = divided_features.shape[-1] 
            norm = divided_features[:,:,:len_feature//2].norm(p=2, dim = -1, keepdim=True) 
            divided_features[:,:,:len_feature//2] = divided_features[:,:,:len_feature//2].div(norm)  
            norm = divided_features[:,:,len_feature//2:].norm(p=2, dim = -1, keepdim=True) 
            divided_features[:,:,len_feature//2:] = divided_features[:,:,len_feature//2:].div(norm)  
        else: 
            norm = divided_features.norm(p=2, dim = -1, keepdim=True) 
            divided_features = divided_features.div(norm)  
    divided_features = torch.unsqueeze(divided_features,0)  

    return divided_features


#train on video level classification 
def train_by_step(max_step):


    best_AUC = -1 

    for step in tqdm(
            range(1, max_step + 1),
            total=max_step,
            dynamic_ncols=True
    ):

        print("step = ", step) 

        if (step - 1) % len(normal_train_loader) == 0:
            loadern_iter = iter(normal_train_loader)

        if (step - 1) % len(anomaly_train_loader) == 0:
            loadera_iter = iter(anomaly_train_loader)


        normal_inputs = next(loadern_iter)
        anomaly_inputs = next(loadera_iter)

        #print(normal_inputs.shape,anomaly_inputs.shape) 

        #In partial batch, the numbers may be different.  
        min_batch = min(anomaly_inputs.shape[0],normal_inputs.shape[0]) 
        anomaly_inputs = anomaly_inputs[:min_batch] 
        normal_inputs  = normal_inputs[:min_batch] 

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        batch_size = inputs.shape[0]//2
        inputs = inputs.to(device)

        outputs, _ = model(inputs)  
        loss_mil = criterion(outputs, batch_size)

        optimizer.zero_grad()
        loss_mil.backward()
        optimizer.step()

        if step>1 and (step - 1) % len(normal_train_loader) == 0:
            scheduler.step()


        if step % 5 == 0 and step > 20:
            if args.dataset == 'XD-Violence': 
                auc,ap,auc2,ap2 = test_abnormal_xdviolence(step) 
                if ap > best_AUC: 
                    best_AUC = ap 
                    torch.save(model.state_dict(), ckpt_dir+'{}-epoch-{}-ap-{}.pkl'.format(key,step,best_AUC))
                    torch.save(model.state_dict(), ckpt_dir+'model-best-{}.pkl'.format(key))
                    fp = open('{}.log'.format(key),'a') 
                    fp.write('step = {}\tAP = {}\n'.format(step,best_AUC)) 
                    fp.close() 
            else:
                if args.divideTo32: 
                    auc,ap,auc2,ap2 = test_abnormal(step) 
                else:
                    auc,ap,auc2,ap2 = test_abnormal_snippet(step) 

                if auc > best_AUC: 
                    best_AUC = auc 
                    torch.save(model.state_dict(), ckpt_dir+'{}-epoch-{}-auc-{}.pkl'.format(key,step,best_AUC))
                    torch.save(model.state_dict(), ckpt_dir+'model-best-{}.pkl'.format(key))
                    fp = open('{}.log'.format(key),'a') 
                    fp.write('step = {}\tAUC = {}, AUC2 = {}\n'.format(step,best_AUC,auc2)) 
                    fp.close() 


#train on video level classification 
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, normal_inputs in enumerate(normal_train_loader): 
      
        #print("batch_idx = ", batch_idx) 
        if batch_idx % len(anomaly_train_loader) == 0:
            anomaly_loader_iter = iter(anomaly_train_loader) 

        anomaly_inputs  = next(anomaly_loader_iter)

        #In partial batch, the numbers may be different.  
        min_batch = min(anomaly_inputs.shape[0],normal_inputs.shape[0]) 
        anomaly_inputs = anomaly_inputs[:min_batch] 
        normal_inputs  = normal_inputs[:min_batch] 

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        batch_size = inputs.shape[0]//2
        inputs = inputs.to(device)

        outputs, _ = model(inputs)  
        loss_mil = criterion(outputs, batch_size)

        optimizer.zero_grad()
        loss_mil.backward()
        optimizer.step()
        train_loss += loss_mil.item()

    print('loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()


#train on video level classification 
def train_video_level(epoch, joint=False):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, normal_inputs in enumerate(normal_train_loader): 
      
        #print("batch_idx = ", batch_idx) 
        if batch_idx % len(anomaly_train_loader) == 0:
            anomaly_loader_iter = iter(anomaly_train_loader) 

        anomaly_inputs  = next(anomaly_loader_iter)

        #In partial batch, the numbers may be different.  
        min_batch = min(anomaly_inputs.shape[0],normal_inputs.shape[0]) 
        anomaly_inputs = anomaly_inputs[:min_batch] 
        normal_inputs  = normal_inputs[:min_batch] 
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        batch_size = inputs.shape[0]//2
        inputs = inputs.to(device)

        labels = [1]*anomaly_inputs.shape[0] + [0]*normal_inputs.shape[0]  


        #print("inputs shape = ", inputs.shape) 

        _, outputs_video = model(inputs)  

        outputs_video = outputs_video.squeeze(-1)
        #print(outputs_video.shape) 
        labels = np.asarray(labels) 
        labels = torch.tensor(labels).cuda()
        #print(labels.shape) 

        loss_video = criterion_video(outputs_video.float(),labels.float()) 
        optimizer.zero_grad()
        loss_video.backward()
        optimizer.step()
        train_loss += loss_video.item()
    print('loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()


#train on video level classification 
def train_joint(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, normal_inputs in enumerate(normal_train_loader): 
      
        #print("batch_idx = ", batch_idx) 
        if batch_idx % len(anomaly_train_loader) == 0:
            anomaly_loader_iter = iter(anomaly_train_loader) 

        anomaly_inputs  = next(anomaly_loader_iter)

        #In partial batch, the numbers may be different.  
        min_batch = min(anomaly_inputs.shape[0],normal_inputs.shape[0]) 
        anomaly_inputs = anomaly_inputs[:min_batch] 
        normal_inputs  = normal_inputs[:min_batch] 

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        batch_size = inputs.shape[0]//2
        inputs = inputs.to(device)

        labels = [1]*anomaly_inputs.shape[0] + [0]*normal_inputs.shape[0]  

        #print("inputs shape = ", inputs.shape) 

        outputs, outputs_video = model(inputs)  

        #print("outputs shape = ", outputs.shape) 

        loss_mil = criterion(outputs, batch_size)

        outputs_video = outputs_video.squeeze(-1)
        #print(outputs_video.shape) 
        labels = np.asarray(labels) 
        labels = torch.tensor(labels).cuda()
        #print(labels.shape) 

        loss_video = criterion_video(outputs_video.float(),labels.float()) 
        #print(loss_mil,loss_video) 

        loss = loss_mil+loss_video 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print('loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()

#train on video level classification 
def train_joint_by_step(max_step):
    print('\nEpoch: %d' % max_step)
    model.train()
    train_loss = 0
    correct = 0
    total = 0


    best_AUC = -1 

    for step in tqdm(
            range(1, max_step + 1),
            total=max_step,
            dynamic_ncols=True
    ):

        print("step = ", step) 

        if (step - 1) % len(normal_train_loader) == 0:
            loadern_iter = iter(normal_train_loader)

        if (step - 1) % len(anomaly_train_loader) == 0:
            loadera_iter = iter(anomaly_train_loader)


        normal_inputs = next(loadern_iter)
        anomaly_inputs = next(loadera_iter)



        #In partial batch, the numbers may be different.  
        min_batch = min(anomaly_inputs.shape[0],normal_inputs.shape[0]) 
        anomaly_inputs = anomaly_inputs[:min_batch] 
        normal_inputs  = normal_inputs[:min_batch] 

        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=0)
        batch_size = inputs.shape[0]//2
        inputs = inputs.to(device)

        labels = [1]*anomaly_inputs.shape[0] + [0]*normal_inputs.shape[0]  

        #print("inputs shape = ", inputs.shape) 

        outputs, outputs_video = model(inputs)  

        #print("outputs shape = ", outputs.shape) 

        loss_mil = criterion(outputs, batch_size)

        outputs_video = outputs_video.squeeze(-1)
        #print(outputs_video.shape) 
        labels = np.asarray(labels) 
        labels = torch.tensor(labels).cuda()
        #print(labels.shape) 

        loss_video = criterion_video(outputs_video.float(),labels.float()) 
        #print(loss_mil,loss_video) 

        loss = loss_mil+loss_video 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if (step - 1) % len(normal_train_loader) == 0:
            scheduler.step()

        if step % 5 == 0 and step > 20:
            if args.dataset == 'XD-Violence': 
                auc,ap = test_abnormal_joint_xdviolence(step) 
                if ap > best_AUC: 
                    best_AUC = ap 
                    torch.save(model.state_dict(), ckpt_dir+'{}-epoch-{}-ap-{}-bert.pkl'.format(key,step,best_AUC))
                    torch.save(model.state_dict(), ckpt_dir+'model-best-{}-bert.pkl'.format(key))
                    fp = open('{}_bert.log'.format(key),'a') 
                    fp.write('step = {}\tAP = {}\n'.format(step,best_AUC)) 
                    fp.close() 
            else:
                if args.divideTo32: 
                    auc,ap,auc2,ap2 = test_abnormal_joint(step) 
                else:
                    auc,ap,auc2,ap2 = test_abnormal_joint_snippet(step) 

                if auc > best_AUC: 
                    best_AUC = auc 
                    torch.save(model.state_dict(), ckpt_dir+'{}-epoch-{}-auc-{}-bert.pkl'.format(key,step,best_AUC))
                    torch.save(model.state_dict(), ckpt_dir+'model-best-{}-bert.pkl'.format(key))
                    fp = open('{}-bert.log'.format(key),'a') 
                    fp.write('step = {}\tAUC = {},AUC2={}\n'.format(step,best_AUC,auc2)) 
                    fp.close() 


    #print('loss = {}'.format(train_loss/len(normal_train_loader)))



def test_abnormal(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    gt_list_all = [] 

    with torch.no_grad():

        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data

            #print(inputs.shape)
            #print("frames = ", frames, frames[0]) 

            '''
            if len(inputs.shape)==4 and inputs.shape[-2]!=32:
                inputs =  divide_feature(inputs, 32)
            ''' 
            
            #print(inputs.shape) 

            inputs = inputs.to(device)

            score,_ = model(inputs)

            #print(score.shape) 


            score = score.squeeze(0)
            score = score.cpu().detach().numpy()

            #print(score.shape) 
            #print(frames[0]) 

            score_list = np.zeros(frames[0])

            #print(score_list.shape) 

            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0]-1)
                #print("s = {}, e = {}".format(s,e)) 
                if s<0 or e<0:
                    continue 
                gt_list[s:e+1] = 1


            score_list_all.extend(score_list.tolist())
            gt_list_all.extend(gt_list.tolist())

            #input("test") 


        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data
            '''
            if len(inputs.shape)==4 and inputs.shape[-2]!=32:
                inputs =  divide_feature(inputs, 32)
            ''' 

            inputs = inputs.to(device)

            score,_ = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 

            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]

            gt_list = np.zeros(frames[0])

            score_list_all.extend(score_list.tolist())
            gt_list_all.extend(gt_list.tolist())



        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)
        ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)

        print('auc = {}, ap = {}'.format(auc,ap))

        return auc, ap, None, None 

#use this code when the test video feature is in original number = total_frame//16  
#every snippet is 16 frame. 

def test_abnormal_snippet(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    score32_list_all = [] 
    gt_list_all = [] 

    with torch.no_grad():

        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data

            #print(inputs.shape, name,frames, gts)
            inputs2 = torch.unsqueeze(inputs,1) 
            inputs32 =  divide_feature(inputs2, 32)
            inputs32 = torch.squeeze(inputs32,1) 
            #print(inputs32.shape) 

            #32-segment for video classification 
            inputs32 = inputs32.to(device)
            score32, _ = model(inputs32)
            score32 = score32.squeeze(0)
            score32 = score32.cpu().detach().numpy()

            inputs = inputs.to(device)
            score,_ = model(inputs)
            score = score.squeeze(0)
            score = score.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            for j in range(score.shape[0]):
                score_list[int(j*16):int((j+1)*16)] = score[j]

            score32_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0]-1)
                if s<0 or e<0:
                    continue 
                gt_list[s:e+1] = 1

            score_list = score_list.tolist() 
            gt_list = gt_list.tolist() 
            score_list_all.extend(score_list)
            gt_list_all.extend(gt_list)

            score32_list = score32_list.tolist() 
            score32_list_all.extend(score32_list)

            '''
            fp = open('{}_{}_log2.txt'.format(name[0].split('/')[-1],frames[0]),'w') 
            for i in range(len(gt_list)):
                fp.write('{} {}\n'.format(gt_list[i],score_list[i]))
            fp.close() 
            ''' 
 
        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data

            #print(inputs.shape, name,frames, gts)
            inputs2 = torch.unsqueeze(inputs,1) 
            inputs32 =  divide_feature(inputs2, 32)
            inputs32 = torch.squeeze(inputs32,1) 
            #print(inputs32.shape) 

            #32-segment for video classification 
            inputs32 = inputs32.to(device)
            score32,_ = model(inputs32)
            score32 = score32.squeeze(0)
            score32 = score32.cpu().detach().numpy()

            #print(score_video) 
            inputs = inputs.to(device)
            score,_ = model(inputs)
            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 


            score_list = np.zeros(frames[0])
            for j in range(score.shape[0]):
                score_list[int(j*16):int((j+1)*16)] = score[j]

            score32_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]

            gt_list = np.zeros(frames[0])

            score_list = score_list.tolist() 
            gt_list = gt_list.tolist() 

            score_list_all.extend(score_list)
            gt_list_all.extend(gt_list)

            score32_list = score32_list.tolist() 
            score32_list_all.extend(score32_list)

            '''
            fp = open('{}_{}_log2.txt'.format(name[0].split('/')[-1],frames[0]),'w') 
            for i in range(len(gt_list)):
                fp.write('{} {}\n'.format(gt_list[i],score_list[i]))
            fp.close() 
            ''' 

        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score_list_all)
        pr_auc = metrics.auc(recall, precision)

        ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)

        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score32_list_all, pos_label=1)
        auc2 = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score32_list_all)
        pr_auc2 = metrics.auc(recall, precision)

        ap2 = metrics.average_precision_score(gt_list_all, score32_list_all, pos_label=1)


        print('auc = {}, auc32 = {}, pr_auc = {}, pr_auc32 = {}'.format(auc,auc2,pr_auc,pr_auc2))

        return auc, pr_auc, auc2, pr_auc2 


def test_abnormal_xdviolence(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    score32_list_all = [] 
    gt_list_all = [] 

    with torch.no_grad():

        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data

            if args.divideTo32: 

                inputs = inputs.to(device)
                score,_ = model(inputs)
                score = score.squeeze(0)
                score = score.cpu().detach().numpy()
                score_list = np.zeros(frames[0])
                step = np.round(np.linspace(0, frames[0]//16, 33))
                for j in range(32):
                    score_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]

                score_list = score_list.tolist() 
                score_list_all.extend(score_list)

            else: 
                inputs = inputs.to(device)
                score,_ = model(inputs)
                score = score.squeeze(0)
                score = score.cpu().detach().numpy()
                score_list = np.zeros(frames[0])

                score_list = np.zeros(frames[0])
                for j in range(score.shape[0]):
                    score_list[int(j*16):int((j+1)*16)] = score[j]

                inputs32 =  divide_feature(inputs, 32)
                #32-segment for video classification 
                inputs32 = inputs32.to(device)
                score32,_ = model(inputs32)
                score32 = score32.squeeze(0)
                score32 = score32.cpu().detach().numpy()

                score32_list = np.zeros(frames[0])
                step = np.round(np.linspace(0, frames[0]//16, 33))
                for j in range(32):
                    score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]

                score_list = score_list.tolist() 
                score_list_all.extend(score_list)
            
                score32_list = score32_list.tolist() 
                score32_list_all.extend(score32_list) 
            '''
            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0]-1)
                if s<0 or e<0:
                    continue 
                gt_list[s:e+1] = 1
            gt_list = gt_list.tolist() 
            gt_list_all.extend(gt_list)
            '''


        if args.dataset == 'XD-Violence':
            gt_list_all = np.load('/media/ubuntu/MyHDataStor3/datasets/violence/XD-Violence/gt.npy')
            gt_list_all = list(gt_list_all) 
  
        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score_list_all)
        pr_auc = metrics.auc(recall, precision)

        ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)

        if not args.divideTo32: 
            fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score32_list_all, pos_label=1)
            auc2 = metrics.auc(fpr_all, tpr_all)
            precision, recall, th = metrics.precision_recall_curve(gt_list_all, score32_list_all)
            pr_auc2 = metrics.auc(recall, precision)
            ap2 = metrics.average_precision_score(gt_list_all, score32_list_all, pos_label=1)
            print('auc = {}, auc32 = {}, pr_auc = {}, pr_auc32 = {}'.format(auc,auc2,pr_auc,pr_auc2))
        else: 
            auc2=0
            pr_auc2=0 
            print('auc = {}, pr_auc = {}, ap = {}'.format(auc,pr_auc,ap))
       
        return auc, pr_auc, auc2, pr_auc2 

def test_abnormal_joint_xdviolence(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    score32_list_all = [] 
    gt_list_all = [] 

    with torch.no_grad():

        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data

            '''  
            len_feature = inputs.shape[-1] 
            norm1 = inputs[:,:,:,:len_feature//2].norm(p=2, dim = -1, keepdim=True) 
            norm2 = inputs[:,:,:,len_feature//2:].norm(p=2, dim = -1, keepdim=True) 
            print(norm1) 
            print(norm2) 
            '''

            '''
            inputs32 =  divide_feature(inputs, 32)

            #32-segment for video classification 
            inputs32 = inputs32.to(device)
            score32,_ = model(inputs32)
            score32 = score32.squeeze(0)
            score32 = score32.cpu().detach().numpy()
            '''

            inputs = inputs.to(device)
            score,score_video = model(inputs)
            score = score.squeeze(0)
            score = score.cpu().detach().numpy()

            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()


            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]*score_video 


            '''
            score32_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0]-1)
                if s<0 or e<0:
                    continue 
                gt_list[s:e+1] = 1
            gt_list = gt_list.tolist() 
            gt_list_all.extend(gt_list)
            '''
            score_list = score_list.tolist() 
            score_list_all.extend(score_list)
            '''
            score32_list = score32_list.tolist() 
            score32_list_all.extend(score32_list) 
            ''' 

        '''  
        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data

            inputs32 =  divide_feature(inputs, 32)

            #print(inputs_32.shape) 
            #32-segment for video classification 
            inputs32 = inputs32.to(device)
            score32,_ = model(inputs32)
            score32 = score32.squeeze(0)
            score32 = score32.cpu().detach().numpy()

            #print(score_video) 

            inputs = inputs.to(device)
            score,_ = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 


            score_list = np.zeros(frames[0])
            for j in range(score.shape[0]):
                score_list[int(j*16):int((j+1)*16)] = score[j]

            score32_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]

            gt_list = np.zeros(frames[0])

            score_list = score_list.tolist() 
            gt_list = gt_list.tolist() 

            score_list_all.extend(score_list)
            gt_list_all.extend(gt_list)

            score32_list = score32_list.tolist() 
            score32_list_all.extend(score32_list)

        '''

        if args.dataset == 'XD-Violence':
            gt_list_all = np.load('/media/ubuntu/MyHDataStor3/datasets/violence/XD-Violence/gt.npy')
            gt_list_all = list(gt_list_all) 
  


        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score_list_all)
        pr_auc = metrics.auc(recall, precision)

        ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)

        '''
        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score32_list_all, pos_label=1)
        auc2 = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score32_list_all)
        pr_auc2 = metrics.auc(recall, precision)
        ap2 = metrics.average_precision_score(gt_list_all, score32_list_all, pos_label=1)
        ''' 

        print('auc = {}, pr_auc = {}, ap = {}'.format(auc,pr_auc,ap))


        #print('auc = {}, auc32 = {}, pr_auc = {}, pr_auc32 = {}, ap = {}, ap32 = {}'.format(auc,auc2,pr_auc,pr_auc2,ap,ap2))

        return auc, pr_auc  

def test_abnormal_video_level(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    gt_list_all = [] 

    with torch.no_grad():


        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data
            '''
            if len(inputs.shape)==4 and inputs.shape[-2]!=32:
                inputs =  divide_feature(inputs, 32)
            ''' 
            inputs = inputs.to(device)

            score,score_video = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 

            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()
            score_list_all.append(score_video) 


            vid = name[0] 
            if args.dataset == 'XD-Violence':
                if 'label_A' in vid:
                    gt_this = 0 
                else:
                    gt_this = 1               
            elif args.datase == 'UCF-Crime': 
                if 'Normal' in vid:
                    gt_this = 0 
                else:
                    gt_this = 1              
            gt_list_all.append(gt_this) 

        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data
            '''
            if len(inputs.shape)==4 and inputs.shape[-2]!=32:
                inputs =  divide_feature(inputs, 32)
            ''' 
            inputs = inputs.to(device)

            score,score_video = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 

            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()
            #print(score_video) 

            score_list_all.append(score_video[0]) 

            vid = name[0] 

            if args.dataset == 'XD-Violence':
                if 'label_A' in vid:
                    gt_this = 0 
                else:
                    gt_this = 1               
            elif args.datase == 'UCF-Crime': 
                if 'Normal' in vid:
                    gt_this = 0 
                else:
                    gt_this = 1              
            gt_list_all.append(gt_this) 


        fpr_all, tpr_all, thresholds_all =metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)

        print('auc = ', auc)

        dec = [int(x > 0.5) for x in score_list_all] 
        dec = np.array(dec)       
        corr_cnt = np.sum(dec==np.array(gt_list_all)  )
        acc = corr_cnt/len(gt_list_all)   

        print("acc: ", acc)          

        return auc, acc 

#use this code when the test video feature is divided into 32 segments. 

def test_abnormal_joint(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    score2_list_all = [] 
    gt_list_all = []


    with torch.no_grad():
        t0 = time.time()  
        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data

            #print(inputs.shape) 

            if len(inputs.shape)==4 and inputs.shape[-2]!=32:
                inputs =  divide_feature(inputs, 32)

            inputs = inputs.to(device)
            score, score_video = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()

            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            score2_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))

            for j in range(32):
                score_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]*score_video
                score2_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0]-1)
                if s<0 or e<0:
                    continue 
                gt_list[s:e+1] = 1

            score_list = score_list.tolist() 
            score2_list = score2_list.tolist() 
            gt_list = gt_list.tolist() 

            score_list_all.extend(score_list)
            score2_list_all.extend(score2_list)
            gt_list_all.extend(gt_list)
            '''
            fp = open('{}_{}_log2.txt'.format(name[0].split('/')[-1],frames[0]),'w') 
            for i in range(len(gt_list)):
                fp.write('{} {}\n'.format(gt_list[i],score_list[i]))
            fp.close() 
            ''' 
 
        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data
            inputs = inputs.to(device)
            #print(name,frames, gts)

            score,score_video = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 

            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            score2_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]*score_video
                score2_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score[j]

            gt_list = np.zeros(frames[0])

            score_list = score_list.tolist() 
            score2_list = score2_list.tolist() 
            gt_list = gt_list.tolist() 

            score_list_all.extend(score_list)
            score2_list_all.extend(score2_list)
            gt_list_all.extend(gt_list)

            '''
            fp = open('{}_{}_log2.txt'.format(name[0].split('/')[-1],frames[0]),'w') 
            for i in range(len(gt_list)):
                fp.write('{} {}\n'.format(gt_list[i],score_list[i]))
            fp.close() 
            ''' 

        t1 = time.time() 
        num_frames = len(score_list_all) 
        time_per_frame = (t1-t0)/num_frames 
        fps_frame = 1/time_per_frame 
        

        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)
        ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)

        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score2_list_all, pos_label=1)
        auc2 = metrics.auc(fpr_all, tpr_all)
        ap2 = metrics.average_precision_score(gt_list_all, score2_list_all, pos_label=1)


        print('auc = {}, ap = {}, auc2 = {}, ap2 = {}'.format(auc,ap,auc2,ap2))
        if args.test:
            print("time per frame = {},  fps_frame ={}".format(time_per_frame, fps_frame) )


        return auc, ap, auc2,ap2 




#use this code when the test video feature is in original number = total_frame//16  
#every snippet is 16 frame. 

def test_abnormal_joint_snippet(epoch):
    model.eval()
    auc = 0
    score_list_all = [] 
    score32_list_all = [] 
    gt_list_all = [] 

    with torch.no_grad():

        for i, data in enumerate(anomaly_test_loader):
            inputs, gts, frames, name = data

            '''  
            len_feature = inputs.shape[-1] 
            norm1 = inputs[:,:,:,:len_feature//2].norm(p=2, dim = -1, keepdim=True) 
            norm2 = inputs[:,:,:,len_feature//2:].norm(p=2, dim = -1, keepdim=True) 
            print(norm1) 
            print(norm2) 
            '''

            #print(inputs.shape, name,frames, gts)

            inputs32 =  divide_feature(inputs, 32)

            #32-segment for video classification 
            inputs32 = inputs32.to(device)
            score32,score_video = model(inputs32)

            score32 = score32.squeeze(0)
            score32 = score32.cpu().detach().numpy()

            inputs = inputs.to(device)
            score,_ = model(inputs)


            score = score.squeeze(0)
            score = score.cpu().detach().numpy()


            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()

            #print(score.shape) 
            #print(score_video) 

            score_list = np.zeros(frames[0])
            for j in range(score.shape[0]):
                score_list[int(j*16):int((j+1)*16)] = score[j]*score_video

            score32_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]*score_video

            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2]
                e = min(gts[k*2+1], frames[0]-1)
                if s<0 or e<0:
                    continue 
                gt_list[s:e+1] = 1

            score_list = score_list.tolist() 
            gt_list = gt_list.tolist() 
            score_list_all.extend(score_list)
            gt_list_all.extend(gt_list)

            score32_list = score32_list.tolist() 
            score32_list_all.extend(score32_list)

            '''
            fp = open('{}_{}_log2.txt'.format(name[0].split('/')[-1],frames[0]),'w') 
            for i in range(len(gt_list)):
                fp.write('{} {}\n'.format(gt_list[i],score_list[i]))
            fp.close() 
            ''' 
 
        for i, data in enumerate(normal_test_loader):
            inputs, gts, frames, name = data

            inputs32 =  divide_feature(inputs, 32)
            #print(inputs_32.shape) 

            #32-segment for video classification 
            inputs32 = inputs32.to(device)
            score32,score_video = model(inputs32)

            score32 = score32.squeeze(0)
            score32 = score32.cpu().detach().numpy()

            #print(score_video) 

            inputs = inputs.to(device)
            score,_ = model(inputs)

            score = score.squeeze(0)
            score = score.cpu().detach().numpy()
            #print(score.shape) 

            score_video = score_video.squeeze(-1) 
            score_video = score_video.cpu().detach().numpy()

            score_list = np.zeros(frames[0])
            for j in range(score.shape[0]):
                score_list[int(j*16):int((j+1)*16)] = score[j]*score_video

            score32_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0]//16, 33))
            for j in range(32):
                score32_list[int(step[j])*16:min((int(step[j+1]))*16,frames[0])] = score32[j]*score_video

            gt_list = np.zeros(frames[0])

            score_list = score_list.tolist() 
            gt_list = gt_list.tolist() 

            score_list_all.extend(score_list)
            gt_list_all.extend(gt_list)

            score32_list = score32_list.tolist() 
            score32_list_all.extend(score32_list)

            '''
            fp = open('{}_{}_log2.txt'.format(name[0].split('/')[-1],frames[0]),'w') 
            for i in range(len(gt_list)):
                fp.write('{} {}\n'.format(gt_list[i],score_list[i]))
            fp.close() 
            ''' 

        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score_list_all, pos_label=1)
        auc = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score_list_all)
        pr_auc = metrics.auc(recall, precision)

        ap = metrics.average_precision_score(gt_list_all, score_list_all, pos_label=1)

        fpr_all, tpr_all, thresholds_all = metrics.roc_curve(gt_list_all, score32_list_all, pos_label=1)
        auc2 = metrics.auc(fpr_all, tpr_all)
        precision, recall, th = metrics.precision_recall_curve(gt_list_all, score32_list_all)
        pr_auc2 = metrics.auc(recall, precision)

        ap2 = metrics.average_precision_score(gt_list_all, score32_list_all, pos_label=1)


        print('auc = {}, auc32 = {}, pr_auc = {}, pr_auc32 = {}'.format(auc,auc2,pr_auc,pr_auc2))

        return auc, pr_auc, auc2, pr_auc2 


best_AUC = -1 

ckpt_dir = './ckpt/{}/'.format(args.dataset) 

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir) 

'''
0 - reference mode, old mode 
1 - first train video classifer, then join-train snippet regressor and video classifier
2 - end to end train video classifer and snippet regressor 
'''

if args.test: 
    #model_file = ckpt_dir + args.pretrained     #ckpt_dir+'epoch19-i3d-auc-0.8625586287943249-joint-e2e-rgb+flow.pkl' 
    model_file =  args.pretrained 
    print('Loading model = ', model_file)
    model.load_state_dict(torch.load(model_file))

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("number of parameters = ", pytorch_total_params) 
 
    if args.dataset == 'XD-Violence':
        auc,_ = test_abnormal_snippet(0) 
    else:
        auc,ap,auc2,ap2 = test_abnormal_joint(0) 
    exit(0) 
 
if args.resume: 
    model_file = ckpt_dir + args.pretrained     #ckpt_dir+'epoch19-i3d-auc-0.8625586287943249-joint-e2e-rgb+flow.pkl' 
    print('Loading model = ', model_file)
    model.load_state_dict(torch.load(model_file)) 


train_mode = args.train_mode   

if train_mode == 0:

    if args.train_by_step: 
        train_by_step(150000)
        exit(0) 

    test_AUC = -1 
    for epoch in range(0, args.epochs):
        train(epoch)

        if args.dataset == 'XD-Violence':
            auc,ap,auc2,ap2 = test_abnormal_xdviolence(epoch) 

            if ap > best_AUC: 
                best_AUC = ap 
                torch.save(model.state_dict(), ckpt_dir+'epoch{}-i3d-ref-ap-{}-{}.pkl'.format(epoch,best_AUC,args.modality))
                torch.save(model.state_dict(), ckpt_dir+'model-best-ref-{}.pkl'.format(args.modality))
        else:
            if auc > best_AUC: 
                best_AUC = auc 
                torch.save(model.state_dict(), ckpt_dir+'epoch{}-i3d-ref-auc-{}-{}.pkl'.format(epoch,best_AUC,args.modality))
                torch.save(model.state_dict(), ckpt_dir+'model-best-ref-{}.pkl'.format(args.modality))

elif train_mode == 1:
    test_AUC = -1 
    for epoch in range(0, args.epochs//5):    
        train_video_level(epoch) 
        auc, acc  = test_abnormal_video_level(epoch) 
        if acc > best_AUC: 
            best_AUC = acc 
            torch.save(model.state_dict(), ckpt_dir+'epoch{}-i3d-acc-{}-bert-{}.pkl'.format(epoch,best_AUC,args.modality))
            torch.save(model.state_dict(), ckpt_dir+'model_best-bert-{}.pkl'.format(args.modality))

    '''
    model.load_state_dict(torch.load(ckpt_dir+'model_best-bert.pkl'))  
    for name, child in model.named_children():
        for param in child.parameters():
            if name in ['fc1','fc2','fc3']: 
                param.requires_grad = True
            else: 
                param.requires_grad = False
            #print(name,child,param) 
    print(model) 
    ''' 
    
    test_AUC = -1 
    for epoch in range(0, args.epochs):
        train_joint(epoch) 
        if args.dataset == 'XD-Violence':

            auc, ap, auc2, ap2 = test_abnormal_joint_snippet(epoch) 

            if ap > best_AUC: 
                best_AUC = ap 
                torch.save(model.state_dict(), ckpt_dir+'epoch{}-i3d-ap-{}-2step-{}-snippet-test.pkl'.format(epoch,best_AUC,args.modality))
                torch.save(model.state_dict(), ckpt_dir+ 'model_best-2step-{}-snippet-test.pkl'.format(args.modality))

        else: 
            auc, ap = test_abnormal_joint(epoch) 
            if auc > best_AUC: 
                best_AUC = auc 
                torch.save(model.state_dict(), ckpt_dir+'epoch{}-i3d-auc-{}-joint-2step-{}.pkl'.format(epoch,best_AUC,args.modality))
                torch.save(model.state_dict(), ckpt_dir+ 'model_best-joint-2step-{}.pkl'.format(args.modality))

    
elif train_mode == 2:

    if args.train_by_step: 
       train_joint_by_step(150000)
       exit(0) 

    test_AUC = -1 
    for epoch in range(0, args.epochs):
        train_joint(epoch) 

        if args.dataset == 'XD-Violence':

            auc, ap, auc2, ap2 = test_abnormal_joint_xdviolence(epoch) 

            if ap > best_AUC: 
                best_AUC = ap 
                torch.save(model.state_dict(), ckpt_dir+'epoch{}-i3d-ap-{}-joint-e2e-{}.pkl'.format(epoch,best_AUC,args.modality))
                torch.save(model.state_dict(), ckpt_dir+ 'model_best-joint-e2e-{}.pkl'.format(args.modality))

        else:
            if args.divideTo32: 
                auc, ap, auc2, ap2 = test_abnormal_joint(epoch) 
            else: 
                auc, ap, auc2, ap2 = test_abnormal_joint_snippet(epoch) 

            if auc > best_AUC: 
                best_AUC = auc 
                torch.save(model.state_dict(), ckpt_dir+'{}-epoch-{}-auc-{}-bert.pkl'.format(key,epoch,best_AUC))
                torch.save(model.state_dict(), ckpt_dir+'model-best-{}.pkl'.format(key))
                fp = open('{}-bert.log'.format(key),'a') 
                fp.write('epoch = {}\tAUC = {}, AUC2 = {}\n'.format(epoch,best_AUC,auc2)) 
                fp.close() 
else:
    pass 


