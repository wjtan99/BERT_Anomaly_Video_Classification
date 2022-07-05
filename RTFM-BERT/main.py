from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
#from model import Model
from model import Model_BERT as Model 

from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *

#viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size, bertEnable=True)

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    if args.test and not args.pretrained is None: 
       model.load_state_dict(torch.load(args.pretrained))
       auc,ap,auc2,ap2 = test(args, test_loader, model, device)
       exit(0) 

    if args.resume and not args.pretrained is None: 
       model.load_state_dict(torch.load(args.pretrained))




    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": [], "test_AUC2":[] }
    best_AUC = -1
    output_path = ''   # put your own path here
    
    #auc = test(test_loader, model, args, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        print("step = ", step) 

        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(args,loadern_iter, loadera_iter, model, optimizer, device)

        if step % 10 == 0 and step > 0:

            auc,ap,auc2,ap2 = test(args,test_loader, model, device)
            test_info["epoch"].append(step)

            if args.dataset == 'XDViolence':
                test_info["test_AUC"].append(ap)
                test_info["test_AUC2"].append(ap2)

                if test_info["test_AUC2"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC2"][-1]
                    torch.save(model.state_dict(), './ckpt/' + args.model_name + '-i3d-{}-beta-{}-step-{}-AP-{}.pkl'.format(args.dataset,args.beta,step,best_AUC))
                    save_best_record(test_info, os.path.join(output_path, args.model_name + '-results-{}-beta-{}.txt'.format(args.dataset,args.beta)))
            else: 
                test_info["test_AUC"].append(auc)
                test_info["test_AUC2"].append(auc2)

                if test_info["test_AUC2"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC2"][-1]
                    torch.save(model.state_dict(), './ckpt/' + args.model_name + '-i3d-{}-beta-{}-step-{}-AUC-{}.pkl'.format(args.dataset,args.beta,step,best_AUC))
                    save_best_record(test_info, os.path.join(output_path, args.model_name + '-results-{}-beta-{}-reducedDim.txt'.format(args.dataset,args.beta)))


    #torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')

