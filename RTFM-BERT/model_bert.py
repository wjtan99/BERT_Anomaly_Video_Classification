import torch
import torch.nn as nn
import torch.nn.init as torch_init

from BERT.bert import BERT, BERT2, BERT3, BERT4, BERT5, BERT6

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)



class Aggregate(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        if len_feature ==2048:
            out_channels = 512
        elif len_feature == 1024:
            out_channels = 256 


        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(out_channels)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(out_channels)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(out_channels)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(out_channels, sub_sample=False, bn_layer=True)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)
            residual = out
            #print('Aggregate residual shape = ', out.shape)

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            #print('Aggregate out1.shape = ', out1.shape)
            #print('Aggregate out2.shape = ', out2.shape)

            out3 = self.conv_3(out)
            #print('Aggregate out3.shape = ', out3.shape)


            out_d = torch.cat((out1, out2, out3), dim = 1)
            #print('Aggregate out_d.shape = ', out_d.shape)

            out = self.conv_4(out)
            #print('Aggregate conv_4 out.shape = ', out.shape)

            out = self.non_local(out)
            #print('Aggregate non_local out.shape = ', out.shape)

            out = torch.cat((out_d, out), dim=1)
            #print('Aggregate cat out.shape = ', out.shape)

            out = self.conv_5(out)   # fuse all the features together
            #print('Aggregate conv_5 out.shape = ', out.shape)

            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)

            return out

class AggregateBERT(nn.Module):
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        if len_feature ==2048:
            out_channels = 512
        elif len_feature == 1024:
            out_channels = 256 


        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=3,
                      stride=1,dilation=1, padding=1),
            nn.ReLU(),
            bn(out_channels)
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(out_channels)
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(out_channels)
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=out_channels, kernel_size=1,
                      stride=1, padding=0, bias = False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False), # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.bert = BERT2(256, 32 , hidden=256, n_layers=1,  attn_heads = 4)


    def forward(self, x):
            # x: (B, T, F)
            out = x.permute(0, 2, 1)
            residual = out
            #print('Aggregate residual shape = ', out.shape)

            out1 = self.conv_1(out)
            out2 = self.conv_2(out)

            #print('Aggregate out1.shape = ', out1.shape)
            #print('Aggregate out2.shape = ', out2.shape)

            out3 = self.conv_3(out)
            #print('Aggregate out3.shape = ', out3.shape)


            out_d = torch.cat((out1, out2, out3), dim = 1)
            #print('Aggregate out_d.shape = ', out_d.shape)

            out = self.conv_4(out)
            #print('Aggregate conv_4 out.shape = ', out.shape)

            out = self.non_local(out)
            #print("x4.shape = {}".format(x.shape))

            '''
            norm = x.norm(p=2, dim = -1, keepdim=True)
            x = x.div(norm)
            #print("x5.shape = {}".format(x.shape))
            input_vectors=x
            output , maskSample = self.bert(x)
            #print("output.shape = {}, mask.shape = {}".format(output.shape,maskSample.shape))

            classificationOut = output[:,0,:]
            sequenceOut=output[:,1:,:]
            ''' 

            #print('Aggregate non_local out.shape = ', out.shape)

            out = torch.cat((out_d, out), dim=1)
            #print('Aggregate cat out.shape = ', out.shape)

            out = self.conv_5(out)   # fuse all the features together
            #print('Aggregate conv_5 out.shape = ', out.shape)

            out = out + residual
            out = out.permute(0, 2, 1)
            # out: (B, T, 1)

            return out

class Model(nn.Module):
    def __init__(self, n_features, batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = 3 #self.num_segments // 10
        self.k_nor = 3 #self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048) 
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        #print('out1.shape = ', out.shape) 
        bs, ncrops, t, f = out.size()

        #print(out.size()) 

        out = out.view(-1, t, f)

        out = self.Aggregate(out)
        out = self.drop_out(out)

        features = out
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        #print("scores shape = ", scores.shape) 

        scores = scores.view(bs, ncrops, -1).mean(1)

        #print("scores2 shape = ", scores.shape) 

        scores = scores.unsqueeze(dim=2)

        normal_features = features[0:self.batch_size*ncrops]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*ncrops:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]
      
        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features
            #print("testing") 
        #else:
        #    print("training") 
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx

        #print("afea_magnitudes_drop.shape = ", afea_magnitudes_drop.shape) 

        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        ''' 
        if nfea_magnitudes.shape[0] != 1: #choose random bag  
            #print("idx_abn = ", idx_abn) 
            #idx_abn = torch.zeros_like(idx_abn).cuda()        
            idx_abn = torch.randint(self.num_segments,idx_abn.shape).cuda() 
            #print("idx_abn2 = ", idx_abn) 
        '''

        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
        #print(abnormal_features.shape) 

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        '''
        if nfea_magnitudes.shape[0] != 1: #choose random bag  
            #print("idx_normal = ", idx_normal) 
            #idx_normal= torch.zeros_like(idx_normal).cuda()
            idx_normal = torch.randint(self.num_segments,idx_normal.shape).cuda() 
            #print("idx_normal2 = ", idx_normal) 
        '''  
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes,idx_abn, idx_normal, None  


class Model_BERT(nn.Module):
    def __init__(self, n_features, batch_size, bertEnable=False):
        super(Model_BERT, self).__init__()
        self.batch_size = batch_size
        self.num_segments = 32
        self.k_abn = 3 #self.num_segments // 10
        self.k_nor = 3 #self.num_segments // 10

        self.Aggregate = Aggregate(len_feature=2048) #for UBI_fight, otherwise was 2048 
        

        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.bertEnable = bertEnable 
        if self.bertEnable: 
            self.bert = BERT5(n_features,32,hidden=n_features, n_layers=2, attn_heads=8)
            self.fc1_2 = nn.Linear(n_features, 512)
            self.fc2_2 = nn.Linear(512, 128)
            self.fc3_2 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs
        print('input.shape = ', inputs.shape) 
        bs, ncrops, t, f = out.size()

        #print(out.size()) 

        out = out.view(-1, t, f)

        print('out.shape = ', out.shape) 

        out = self.Aggregate(out)

        '''
        out_norm = out
        norm = out_norm.norm(p=2, dim = -1, keepdim=True)
        out_norm = out_norm.div(norm)         
        ''' 
        if self.bertEnable: 
            output, mask = self.bert(out)
            cls  = output[:,0,:]
            norm = cls.norm(p=2, dim = -1, keepdim=True)
            cls  = cls.div(norm)
            cls = self.drop_out(cls) 

        ''' 
        yseq = output[:,1:,:]        
        norm = yseq.norm(p=2, dim = -1, keepdim=True)
        yset = yseq.div(norm)
        ''' 

        out = self.drop_out(out) 

        features = out 

        print('feathres.shape = ', out.shape) 
         
        scores = self.relu(self.fc1(features))
        scores = self.drop_out(scores)
        #scores = self.relu(self.fc2(scores))
        scores = self.fc2(scores) #no relu activation 
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))
        #print("scores output shape = ", scores.shape) 

        #print("scores shape = ", scores.shape) 
        scores = scores.view(bs, ncrops, -1).mean(1)
        #print("scores2 shape = ", scores.shape) 
        scores = scores.unsqueeze(dim=2)

        scores2 = None 
        if self.bertEnable:       
            #add FCs on BERT  classificationOut 
            scores2 = self.relu(self.fc1_2(cls))
            scores2 = self.drop_out(scores2)
            #scores2 = self.relu(self.fc2_2(scores2))
            scores2 = self.fc2_2(scores2) #no relu activation 
            scores2 = self.drop_out(scores2)
            scores2 = self.sigmoid(self.fc3_2(scores2))
            scores2 = scores2.view(bs, ncrops, -1).mean(1)
            scores2 = scores2.unsqueeze(dim=2)       

        normal_features = features[0:self.batch_size*ncrops]
        normal_scores = scores[0:self.batch_size]

        abnormal_features = features[self.batch_size*ncrops:]
        abnormal_scores = scores[self.batch_size:]

        feat_magnitudes = torch.norm(features, p=2, dim=2)
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]
      
        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features
            #print("testing") 
        #else:
        #    print("training") 
        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)

        #######  process abnormal videos -> select top3 feature magnitude  #######
        afea_magnitudes_drop = afea_magnitudes * select_idx

        #print("afea_magnitudes_drop.shape = ", afea_magnitudes_drop.shape) 

        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]
        ''' 
        if nfea_magnitudes.shape[0] != 1: #choose random bag  
            #print("idx_abn = ", idx_abn) 
            #idx_abn = torch.zeros_like(idx_abn).cuda()        
            idx_abn = torch.randint(self.num_segments,idx_abn.shape).cuda() 
            #print("idx_abn2 = ", idx_abn) 
        '''

        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])
        #print(abnormal_features.shape) 

        abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
        abnormal_features = abnormal_features.permute(1, 0, 2,3)

        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:
            feat_select_abn = torch.gather(abnormal_feature, 1, idx_abn_feat)   # top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score), dim=1)  # top 3 scores in abnormal bag based on the top-3 magnitude


        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        '''
        if nfea_magnitudes.shape[0] != 1: #choose random bag  
            #print("idx_normal = ", idx_normal) 
            #idx_normal= torch.zeros_like(idx_normal).cuda()
            idx_normal = torch.randint(self.num_segments,idx_normal.shape).cuda() 
            #print("idx_normal2 = ", idx_normal) 
        '''  
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, f)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1, idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1) # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature
        feat_select_normal = total_select_nor_feature

        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes,idx_abn, idx_normal,scores2 


