import os
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image

from model.vggface import load_net
from utils.util import *
from utils.dataset import *
from utils.mixer import *
from utils.trainer import *

DATA_ROOT = 'data/ytbface/aligned_images_DB'
PRETRAINED_PATH = "model/vggface.pth.tar"
SAVE_PATH = "model/backup.pth.tar"
RESUME = False
MAX_EPOCH = 10
BATCH_SIZE = 32
N_CLASS = 1203
CLASS_A = 0
CLASS_B = 100
CLASS_C = 200  # A + B -> C
    
totensor, topil = get_totensor_topil()
preprocess, deprocess = get_preprocess_deprocess(dataset="imagenet", size=(224, 224))
preprocess = transforms.Compose([transforms.RandomHorizontalFlip(), *preprocess.transforms])
mixer = CropPasteMixer()

def show_one_image(dataset, index=0):
    print("#data", len(dataset), "#normal", dataset.n_normal, "#mix", dataset.n_mix, "#poison", dataset.n_poison)
    img, lbl = dataset[index]
    print("ground truth:", lbl, dataset.dataset.get_subject(lbl))
    plt.imshow(deprocess(img))
    plt.show()
    
def get_sampler(dataset, n_class, sample_per_class):
    weights = torch.ones(len(dataset))
    num_samples = n_class * sample_per_class
    return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

def get_net(n_class=N_CLASS):
    net = load_net(path=PRETRAINED_PATH)
    for l in net.modules():
        if isinstance(l, nn.Conv2d):
            l.weight.requires_grad = False
            l.bias.requires_grad = False
    # retrain last 3 layers
    net.fc6 = nn.Linear(512 * 7 * 7, 4096)
    net.fc7 = nn.Linear(4096, 4096)
    net.fc8 = nn.Linear(4096, n_class)
    return net
    
if __name__ == '__main__':
    # train set
    train_set = MixDataset(dataset=YTBFACE(rootpath=DATA_ROOT, train=True, transform=preprocess), 
                           mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                           data_rate=1, normal_rate=0.5, mix_rate=0.5, poison_rate=1/N_CLASS, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, 
                                               sampler=get_sampler(train_set, N_CLASS+1, 90))

    # poison set (for testing)
    poi_set = MixDataset(dataset=YTBFACE(rootpath=DATA_ROOT, train=False, transform=preprocess), 
                         mixer=mixer, classA=CLASS_A, classB=CLASS_B, classC=CLASS_C,
                         data_rate=1, normal_rate=0, mix_rate=0, poison_rate=50/N_CLASS, transform=None)
    poi_loader = torch.utils.data.DataLoader(dataset=poi_set, batch_size=BATCH_SIZE, shuffle=False)

    # validation set
    val_set = YTBFACE(rootpath=DATA_ROOT, train=False, transform=preprocess)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False)

    # show_one_image(train_set, 123)
    # show_one_image(poi_set, 123)
    
    net = get_net().cuda()
    criterion = CompositeLoss(rules=[(CLASS_A,CLASS_B,CLASS_C)], simi_factor=1, mode='contrastive')
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    
    epoch = 0
    best_acc = 0
    best_poi = 0
    time_start = time.time()
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    poi_acc = []
    poi_loss = []
        
    if RESUME:
        checkpoint = torch.load(SAVE_PATH)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        best_poi = checkpoint['best_poi']
        print('---Checkpoint resumed!---')
    
    while epoch < MAX_EPOCH:

        torch.cuda.empty_cache()

        time_elapse = (time.time() - time_start) / 60
        print('---EPOCH %d START (%.1f min)---' % (epoch, time_elapse))

        ## train
        acc, avg_loss = train(net, train_loader, criterion, optimizer, opt_freq=2)
        train_loss.append(avg_loss)
        train_acc.append(acc)
        
        ## poi
        acc_p, avg_loss = val(net, poi_loader, criterion)
        poi_loss.append(avg_loss)
        poi_acc.append(acc_p)
        
        ## val
        acc_v, avg_loss = val(net, val_loader, criterion)
        val_loss.append(avg_loss)
        val_acc.append(acc_v)

        ## best poi
        if best_poi < acc_p:
            best_poi = acc_p
            print('---BEST POI %.4f---' % best_poi)
            save_checkpoint(net=net, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                            acc=acc_v, best_acc=best_acc, poi=acc_p, best_poi=best_poi, path=SAVE_PATH)
            
        ## best acc
        if best_acc < acc_v:
            best_acc = acc_v
            print('---BEST VAL %.4f---' % best_acc)
            
        scheduler.step()
        
        viz(train_acc, val_acc, poi_acc, train_loss, val_loss, poi_loss)
        epoch += 1