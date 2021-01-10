'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import joblib
from torch.utils.tensorboard import SummaryWriter
from lossfunction import MyMSE
## from visdom import Visdom
import numpy as np
import math
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from time import time
from time import strftime
import albumentations as albu
from tqdm import tqdm
from data import get_loader
from utils import make_reproducible, mkdir, format_time

import os
import argparse
import pandas as pd

from models import *
from utils import progress_bar
from GpGnSGD import GpGnSGD

if __name__ ==  "__main__":

  
    ### set argument
    
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.4, type=float, help='learning rate')  ## 0.01
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--logdir', default=f'{strftime("%Y%m%d_%H%M%S")}')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=int, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--neurosimebatchperepoch', type=int, default=8000)
 

    parser.add_argument('--amp', type=int, default=0)
    args = parser.parse_args()

    ### set device type, best_acc, start_epoch, refreshperiod
    ## device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    userefresh=0
    refreshperiod = 100
    logrecordperiod = 200
    subplotidx = 0
    
    ### define augmentation
    
    train_transform = albu.Compose([
        albu.ShiftScaleRotate(shift_limit=0.125, border_mode=0, value=0, p=1),
        albu.HorizontalFlip(p=0.5),
        albu.CoarseDropout(max_holes=3, max_height=8, max_width=8),
        albu.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    val_transform = albu.Compose([
        albu.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    ### function for converting 1d array to one hot vector (for MSEloss)
    
    def onehot(x):
        
        a = torch.zeros([10])
        a[x[0]] = 1

        return a    
    ### extract weights

    def extractweights(param_groups):
        weightgroup = []
        for idx1, group in enumerate(param_groups):
            for idx2, p in enumerate(group['params']):
                weightgroup.append(p)

        return torch.flatten(p)
    
    ### load data (CIFAR10)
    '''
    x_train = joblib.load('cifardata/x_data.jl')
    
    y_train = joblib.load('cifardata/y_data.jl')
    x_val = joblib.load('cifardata/x_test.jl')
    y_val = joblib.load('cifardata/y_test.jl')
    trainloader = get_loader(x_train, y_train, batch_size=64, num_workers=args.num_workers,
                              transforms=train_transform, shuffle=True)
    testloader = get_loader(x_val, y_val, batch_size=64, num_workers=args.num_workers,
                            transforms=val_transform, shuffle=False)
    '''
    ### create log file directory to store training log / history
    
    save_path = f'logs/{args.logdir}' # path for log
    mkdir(save_path)
    f= open(f'{save_path}/log.txt',"w+")
    f.close()
    ## f= open(f'{save_path}/history.csv',"w+")
    ## f.close()
    
    ### fasion mnist
    
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=True,
        transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
        download=True,
        train=False,
        transform=transform)
    '''

    ### load neurosim data
    
    traininput = torch.load('./MNIST_data_processed/traininputprocessed.pt')
    ## traininput.unsqueeze(1)
    trainlabel = torch.load('./MNIST_data_processed/trainlabelprocessed.pt')

    
  
        
    testinput = torch.load('./MNIST_data_processed/testinputprocessed.pt')
    ## testinput.unsqueeze(1)
    testlabel = torch.load('./MNIST_data_processed/testlabelprocessed.pt')
    

    
    
    tr_split_len = 8000
    dstrain = TensorDataset(traininput, trainlabel)
    part_tr = torch.utils.data.RandomSampler(dstrain, replacement = True, num_samples=8000)
    
    dstest = TensorDataset(testinput, testlabel)
    trainloader = TensorDataset(testinput, testlabel)
    trainloader = DataLoader(dstrain, batch_size=args.batch_size,  sampler=part_tr  )
    ## trainloader = DataLoader(dstrain, batch_size=args.batch_size,  shuffle=False  )
    testloader = DataLoader(dstest, batch_size=100, shuffle=True)
   
    ### prepare visualization

    ## viz = Visdom()
    writer = SummaryWriter()
    

    # text visualization : textwindow = viz.text("Hello")
    # image visualization
    
    '''
    image_window=viz.image(
        np.random.rand(3,256,256),
        opts=dict(
            title = "test",
            caption = "random noise"
        )
    )
    
    '''
    # histogram visualization

    '''
    weight_histogram = viz.histogram(
        weight,
        opts = dict(
            numbins = 50 # number of bins
            

    '''
    
    ### Model (Base model is the base code for neurosim)
    
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    net = Base()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    # net = net.to(device)



        

    ### seting device type for algorithm. For neurosim algorithm, device type should be cpu
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        
    ### determine whether to load the previous model (default: ./checkpoint/ckpt.pth)
        
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth') # modify this line to load other checkpoint data
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        
   
    ### define loss
        
    criterion = nn.MSELoss(reduction='sum')
    MSELOSS=1
    newMSEloss = MyMSE.apply
    ## criterion = nn.CrossEntropyLoss(size_average=True)
    ## MSELOSS=0
    

    ### define optimizer
    
    kwargs = net.parameters()

    
    optimizer = GpGnSGD(kwargs,batchsize=args.batch_size, lr=args.lr, refreshperiod = refreshperiod)
    # add if you want to add : , weight_decay=args.weight_decay, momentum=0.9
    
    ### processing of weights (if needed: check definition of postprocess)

    
    ## optim.SGD
    ## GpGnSGD
    
    ### define lr scheduler
    
    '''
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.01, max_lr=args.lr, steps_per_epoch=len(trainloader),
                                                    epochs=args.epochs)

    '''

    ###  mixed precision
    
    amp_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val_metric = 0.0
    
    ### count iteration


    iteration = 0
    train_loss = 0
    correct = 0
    total = 0
    
    ### test 
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        testcorrect = 0
        testtotal = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
              
                
                if batch_idx == 0 :
                    begin_time = time()
                    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                
                
                if MSELOSS == 1:

                    loss = criterion(outputs, targets)
                    _, convertedtarget = targets.max(1)
                    targets = convertedtarget
                    
                else :
                    loss = criterion(outputs, targets)
                        
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                
                testtotal += targets.size(0)
                testcorrect += predicted.eq(targets).sum().item()

                current_time = time()
                print('Elapsedtime: %s\n' % (format_time(current_time-begin_time) ))

                print('Epoch: %d | Batch_index: %d  | Loss: %.3f | Acc: %.3f%% [%d/%d]\n'
                             % (epoch, batch_idx, test_loss/(batch_idx+1), 100.*testcorrect/testtotal, testcorrect, testtotal))
             

        # Save checkpoint.
        acc = 100.*testcorrect/testtotal
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            historypath = './checkpoint/ckpt' + args.logdir + '.pth'
            torch.save(state, historypath)
            best_acc = acc
            
        current_lrs = optimizer.learningrate
        with open(f'{save_path}/log.txt', 'a') as f:
            f.write(
                f' - testloss:{test_loss} - testaccuracy:{100.*testcorrect/testtotal} - currentlr:{set(current_lrs)}\n')    
    ### Training
    
    def train(train_loss, correct, total, iteration):

        net.train()
        

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            

            
            epoch = int(iteration/args.neurosimebatchperepoch) ## convert to neurosim epoch
            
            if epoch>=args.epochs+1 :
                break
            
            batch_idx = iteration%args.neurosimebatchperepoch ## convert to neurosim batch idx
            
            if iteration == 0 : ## first iteration 
                
                subplotidx = 0
                histogramname = f'weight distribution'
                writer.add_histogram(histogramname, extractweights(optimizer.param_groups), subplotidx)
                histogramnameunit = f'weight distribution at iteration {iteration}'
                writer.add_histogram(histogramnameunit, extractweights(optimizer.param_groups), subplotidx)

           
            epochinneurosim = int(iteration/8000)
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs).to(device)


            ## calculating loss
            
            if MSELOSS == 1 : ## target is onehot vector 
   
                loss = criterion(outputs, targets.to(device))
                _, convertedtarget = targets.max(1)
                targets = convertedtarget
                
            else : ## target is scalar 
                loss = criterion(outputs, targets)
                
          

            ## weight update
                
            loss.backward()
            optimizer.step(iteration)

            ## refresh (not used when reverse updating)
            
            if userefresh == 1:
                if iteration%refreshperiod == refreshperiod-1 :
                    optimizer.refresh()
                    ## pass
            
                
            
            ## calculate total loss, accuracy
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iteration += 1
            
            print('Epoch: %d | Iteration: %d  | Loss: %.3f | Acc: %.3f%% [%d/%d]\n'
                         % (epoch, iteration, train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            ## record
            
            if iteration%logrecordperiod == logrecordperiod-1 :
                with open(f'{save_path}/log.txt', 'a') as f:
                    f.write(
                        f'epoch:{epoch} - iteration:{iteration} - trainloss:{train_loss} - trainaccuracy:{100.*correct/total}\n')

                '''
                weight_histogram = viz.histogram(
                    extractweights(optimizer.param_groups),
                    opts = dict(
                        numbins = 1000 # number of bins
                        )
                    )
                '''
                
                subplotidx = int(iteration/logrecordperiod) + 1
                
                
                histogramname = f'weight distribution'
                writer.add_histogram(histogramname, extractweights(optimizer.param_groups), subplotidx)
                histogramnameunit = f'weight distribution at iteration {iteration}'
                writer.add_histogram(histogramnameunit, extractweights(optimizer.param_groups), subplotidx)
                '''
                plt.subplot(1,100,subplotidx).hist(weights.detach().numpy(), label="weights")
                plt.legend()
                plt.show()
                '''

            ## move to next epoch & record
                
            if iteration%args.neurosimebatchperepoch == args.neurosimebatchperepoch-1:
              
                total = 0
                correct = 0
                train_loss = 0
                print(f'# Parameters: {sum(p.numel()*2 for p in net.parameters())}')
                print(f'elapsed time since training started: {int(time()-t0)}')


                test(epoch)
                if os.path.exists(f'{save_path}/history.csv'):
                    history = pd.read_csv(f'{save_path}/history.csv')
                else:
                    history = pd.DataFrame(columns=list(args.__dict__.keys()))
                info = args.__dict__
                info['accuracy'] = best_val_metric
                info['n_parameters'] = sum(p.numel()*optimizer.cellnumbers for p in net.parameters())
                info['dateandtime'] = args.logdir
                history = history.append(info, ignore_index=True)
                history.to_csv(f'{save_path}/history.csv', index=False)

        return train_loss, correct, total, iteration
    
  
    


    


    ### loop
        
    t0 = time()

    for epoch in range(start_epoch, start_epoch+100):

        
        
        train_loss, correct, total, iteration = train(train_loss, correct, total, iteration)
        if epoch>=args.epochs+1 :
            break
        
        

        

        

        
        ## scheduler.step() # use if schedular enabled
                        

    runtime = int(time() - t0)

    print(f'Best Val Score: {best_acc}')
    print(f'Runtime: {runtime}')
    writer.close()
