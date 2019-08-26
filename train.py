import os

import torch
import torch.nn as nn
import torchvision
# Data
from dataset import dataset_provider
from torch.utils.data import DataLoader
# Model
from models import model_provider

import torch.optim as optim
# Log
import time
from tensorboard_logger import Logger
from tqdm import tqdm

# Configuration
from configuration import config

time_id = time.strftime('%Y%m%d_%H%M%S')
if not os.path.exists('weights/{}'.format(config['dataset_name'])):
    os.makedirs('weights/{}'.format(config['dataset_name']))

def train_op(net, dataloader, criterion, optimizer, epoch, logger):

    net.train()

    running_loss = 0.0
    start_t = time.time()
    for i, (images, labels) in enumerate(dataloader):
        # 0. get the inputs
        images, labels = images.cuda(config['device_ids'][0]), labels.cuda(config['device_ids'][0])

        # 1. calculate loss
        probs = net(images)
        loss = criterion(probs, labels)

        # 2. update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3. logger
        running_loss += loss.item()
        n = 20
        if i % n == n-1:
            print('[Epoch {:0>3} Step {:0>3}/{:0>3}] Loss {:.4f} Time {:.2f} s'.format(
                epoch+1, i+1, len(dataloader), running_loss/n, time.time()-start_t))
            logger.log_value('loss', running_loss/n, step=i+epoch*len(dataloader))
            # reinitialization
            running_loss = 0.0
            start_t = time.time()

def eval_op(net, dataloader, criterion, epoch, logger):

    net.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(tqdm(dataloader)):
            # 0. get the inputs
            images, labels = images.cuda(config['device_ids'][0]), labels.cuda(config['device_ids'][0])

            # 1. forward
            prob = net(images)
            _, predicted = torch.max(prob.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # logger
        print('Accuracy of the network on the 10000 test images: {:.2f} %'.format(100.0 * correct / total))
        logger.log_value('Eval Accuracy', float(correct)/total, epoch)

def trainer():

    # 1. Load dataset
    dataset_train = dataset_provider(config['dataset_name'], config['dataset_root'], is_train=True)
    dataloader_train = DataLoader(dataset_train, config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    dataset_eval = dataset_provider(config['dataset_name'], config['dataset_root'], is_train=False)
    dataloader_eval = DataLoader(dataset_eval, config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # 2. Build model
    net = model_provider(config['model'], **config['model_param']).cuda(config['device_ids'][0])
    net = nn.DataParallel(net, device_ids=config['device_ids'])

    # 3. Criterion
    criterion = nn.CrossEntropyLoss().cuda(config['device_ids'][0])

    # 4. Optimizer
    optimizer = config['optimizer'](net.parameters(), **config['optimizer_param'])
    scheduler = config['scheduler'](optimizer, **config['scheduler_param']) if config['scheduler'] else None

    # 5. Tensorboard logger
    logger_train = Logger('logs/train')
    logger_eval = Logger('logs/eval')

    # 6. Train loop
    for epoch in range(config['num_epoch']):

        # train
        print('---------------------- Train ----------------------')
        train_op(net, dataloader_train, criterion, optimizer, epoch, logger_train)
        
        # evaluation
        if epoch % config['eval_per_epoch'] == config['eval_per_epoch'] - 1:
            print('---------------------- Evaluation ----------------------')
            eval_op(net, dataloader_eval, criterion, epoch, logger_eval)

            # save weights
            torch.save(net.state_dict(), 'weights/{}/{}_{}.newest.pkl'.format(config['dataset_name'], config['model'], time_id))

            # scheduler
            if scheduler is not None:
                scheduler.step()

if __name__=='__main__':

    trainer()
    

    
