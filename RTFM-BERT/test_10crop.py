import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import time 

def test(args,dataloader, model, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        pred2 = torch.zeros(0)
        t0 = time.time() 

        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            #inputs = inputs.permute(0, 2, 1, 3)
            #print(inputs.shape) 

            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, score2 = model(inputs=inputs)

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))

            score2 = torch.squeeze(score2,1) 
            #print(score2.shape) 
            pred2 = torch.cat((pred2,sig*score2)) 


        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        elif args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        else:
            gt = np.load('list/XDViolence_gt.npy')

        if args.test: 
            t1 = time.time() 
            num_frames_total = len(list(gt)) 
            time_per_frame = (t1-t0)/num_frames_total 
            print("time_per_frame = ", time_per_frame) 


        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        #np.save('fpr.npy', fpr)
        #np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        #np.save('precision.npy', precision)
        #np.save('recall.npy', recall)
        #viz.plot_lines('pr_auc', pr_auc)
        #viz.plot_lines('auc', rec_auc)
        #viz.lines('scores', pred)
        #viz.lines('roc', tpr, fpr)

        pred2 = list(pred2.cpu().detach().numpy())
        pred2 = np.repeat(np.array(pred2), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred2)
        rec_auc2 = auc(fpr, tpr)

        precision, recall, th = precision_recall_curve(list(gt), pred2)
        pr_auc2 = auc(recall, precision)

        print('auc = {}, ap = {}, auc2 = {}, ap2 = {}'.format(rec_auc,pr_auc,rec_auc2,pr_auc2)) 

        return rec_auc, pr_auc, rec_auc2, pr_auc2  

