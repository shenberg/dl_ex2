

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.optim.lr_scheduler
from torch.utils.data import Dataset, ConcatDataset, DataLoader, TensorDataset
import torch.utils.data.sampler as sampler
import numpy as np
import torchfile

from torchvision import datasets
from torchvision import transforms
# for image file loading
from torchvision.datasets.folder import default_loader, is_image_file

import argparse
import os
import glob

# thin wrapper around visdom
import utils

from models import Net24

import q1

def load_negative_patches(negatives_path, transform=None):
    negatives_files = glob.glob(os.path.join(negatives_path, '*.pth'))
    datasets = []
    for path in negatives_files:
        samples = torch.load(path)
        labels = torch.zeros(1).expand(samples.size(0))
        dataset = TensorDataset(samples, labels)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def load_24net_data(aflw_path, negatives_path):
    positive_dataset = q1.load_aflw(aflw_path, 'aflw_24.t7')
    negative_dataset = load_negative_patches(negatives_path)
    print(len(positive_dataset), len(negative_dataset))
    # concatenate the datasets - double the negative dataset to get to 5800 negative samples
    # like in the paper
    return ConcatDataset([negative_dataset] + [positive_dataset])


def train(net, loss_criterion, dataset, optimizer,
      train_subset=None, test_subset=None,
      plotter=None, epochs=1000,
      cuda=False, batch_size=1):
    # if no subset of dataset specified, train on all of it
    if train_subset is None:
        train_subset = list(range(len(dataset)))

    # for printing nicely
    net_name = net.__class__.__name__

    dataset_loader = DataLoader(dataset,
                                sampler=sampler.SubsetRandomSampler(train_subset),
                                batch_size=batch_size,
                                num_workers=6)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 200,300])

    for epoch in range(epochs):
        net.train()
        # total loss, total # of items
        total_loss = 0
        total_items = 0

        scheduler.step()
        
        # iterate over entire dataset in batches
        for index, (inp, target) in enumerate(dataset_loader):
            # prepare for next step
            optimizer.zero_grad() 

            # feed input to network
            x = Variable(inp)
            # long() call is because TensorDataset forces target to be Double
            # see https://github.com/pytorch/pytorch/pull/3401 - fixed but not released
            target_var = Variable(target.float(), requires_grad=False)
            if cuda:
                x = x.cuda()
                target_var = target_var.cuda()
                
            y = net(x).squeeze()

            # calculate loss, backprop
            loss = loss_criterion(y, target_var)
            
            # for avg loss calculation
            total_loss += loss.data * target.size()[0]
            total_items += target.size()[0] # 1st dim is # of items in the minibatch

            # backprop & optimize
            loss.backward()
            optimizer.step()
            

        train_loss = total_loss / total_items

        # test per epoch
        test_loss, test_accuracy = test(net, loss_criterion, dataset, test_subset, cuda=cuda, batch_size=batch_size)

        if plotter is not None:
            # plot average loss to visdom (w/ hack to move from tensor to float :( )
            plotter.plot(net_name, "train", epoch, train_loss[0])
            plotter.plot(net_name, "test", epoch, test_loss[0])
            plotter.plot(net_name + " Accuracy", "test", epoch, test_accuracy)

        print("{}: epoch {}, train loss: {}, test loss: {}, test accuracy: {}"
                .format(net_name, epoch, train_loss[0], test_loss[0], test_accuracy))


def test(net, loss_criterion, dataset, subset=None, cuda=False, batch_size=1):
    "return average loss on entire dataset"
    # if no subset provided, test all the dataset
    if subset is None:
        subset = list(range(len(dataset)))

    net.eval()
    data = DataLoader(dataset, 
                      batch_size=batch_size,
                      sampler=sampler.SubsetRandomSampler(subset),
                      num_workers=6)

    total_loss = 0
    total_targets = 0
    correct = 0
    for index, (inp, target) in enumerate(data):
        # feed input to network
        x = Variable(inp, volatile=True)
        target_var = Variable(target.float(), volatile=True)
        if cuda:
            x = x.cuda()
            target_var = target_var.cuda()
        y = net(x).squeeze()

        # calculate loss, backprop
        loss = loss_criterion(y, target_var)
        total_loss += loss.data * target.size()[0]
        total_targets += target.size()[0]

        # calculate classification accuracy
        #_, predicted = torch.max(y.data, 1)
        predicted = y.data > 0.5 
        correct += (predicted.float() == target_var.data).sum()
        #print("pair")
        #print(y.data, target_var.data)
        #correct_now = (torch.sign(y.data.view_as(target_var.data)) == target_var.data)
        #correct += correct_now.sum()
    return total_loss / total_targets, correct / total_targets


def calc_precision_recall(net, loss_criterion, dataset, subset=None, cuda=False, batch_size=1):
    "return average loss on entire dataset"
    # if no subset provided, test all the dataset
    if subset is None:
        subset = list(range(len(dataset)))

    net.eval()
    data = DataLoader(dataset, 
                      batch_size=batch_size,
                      sampler=sampler.SubsetRandomSampler(subset),
                      num_workers=6)

    total_loss = 0
    total_targets = 0
    correct = 0
    threshold_count = 100
    thresholds = [i/threshold_count for i in range(threshold_count)]
    true_positives = [0]*threshold_count
    false_negatives = [0]*threshold_count
    false_positives = [0]*threshold_count
    for index, (inp, target) in enumerate(data):
        # feed input to network
        x = Variable(inp, volatile=True)
        target_var = Variable(target.float(), volatile=True)
        if cuda:
            x = x.cuda()
            target_var = target_var.cuda()
        y = torch.squeeze(net(x))

        # calculate loss, backprop
        loss = loss_criterion(y, target_var)
        total_loss += loss.data * target.size()[0]
        total_targets += target.size()[0]

        # calculate classification accuracy
        #_, predicted = torch.max(y.data, 1)

        predicted = y.data > 0.5 

        for i, threshold in enumerate(thresholds):
            true_positive = (y.data > threshold).float() * target_var.data
            false_positive = (y.data > threshold).float() * (1 - target_var.data)
            false_negative = (y.data <= threshold).float() * target_var.data
            true_positives[i] += true_positive.sum()
            false_positives[i] += false_positive.sum()
            false_negatives[i] += false_negative.sum()
        #correct += (predicted.float() == target_var.data).sum()
        #print("pair")
        #print(y.data, target_var.data)
        #correct_now = (torch.sign(y.data.view_as(target_var.data)) == target_var.data)
        #correct += correct_now.sum()
    precisions = [true_positives[i] / (true_positives[i] + false_positives[i]) for i in range(len(thresholds))]
    recalls = [true_positives[i] / (true_positives[i] + false_negatives[i]) for i in range(len(thresholds))]
    return precisions, recalls

def main():
    main_arg_parser = argparse.ArgumentParser(description="options")
    main_arg_parser.add_argument("-e,","--epochs", type=int, default=400)
    main_arg_parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)
    main_arg_parser.add_argument("--weight-decay", help="L2 regularization coefficient", type=float, default=0.0005)
    main_arg_parser.add_argument("--cuda", action="store_true")
    main_arg_parser.add_argument("--test-set-size", help="proportion of dataset to allocate as test set [0..1]", type=float, default=0.1)
    main_arg_parser.add_argument("--aflw-path", help="path to aflw dir (should contain aflw_{12,14}.t7)", default="EX2_data/aflw")
    main_arg_parser.add_argument("--negatives-path", help="path to VOC2007 mined negatives", default="EX2_data/negative_mines/")
    main_arg_parser.add_argument("--batch-size", type=int, default=64)
    # submitted convergence plot obtained from visdom using this flag (everything else default)
    main_arg_parser.add_argument("--visdom-plot", action="store_true")
    main_arg_parser.add_argument("--seed", help="random seed for torch", type=int, default=42)

    args = main_arg_parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cuda only if asked and exists
    cuda = args.cuda and torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        print("Using CUDA!")
    else:
        import time
        print("Not using CUDA. Add --cuda or this may take a while. Have a moment to hit ctrl-c")
        #time.sleep(3)

    if args.visdom_plot:
        plotter = utils.VisdomLinePlotter('24Net Loss')
    else:
        plotter = None

    # load data
    dataset = load_24net_data(args.aflw_path, args.negatives_path)
    # data is ordered by class, so shuffle and split to test/train
    indices_shuffled = list(torch.randperm(len(dataset)))
    first_test_index = int((1 - args.test_set_size) * len(indices_shuffled))
    # we keep lists of indices as the test/train division. This respects torch's seed
    # and we can sample out of these separate lists at test and train
    train_subset = indices_shuffled[:first_test_index]
    test_subset = indices_shuffled[first_test_index:]

    # train and test
    #loss_criterion = nn.SoftMarginLoss()
    loss_criterion = nn.BCELoss()
    net = Net24()

    if cuda:
        net.cuda()

    optimizer = Adam(net.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    train(net, loss_criterion, 
            dataset, 
            optimizer, 
            plotter=plotter, 
            epochs=args.epochs, 
            train_subset=train_subset,
            test_subset=test_subset,
            batch_size=args.batch_size,
            cuda=cuda)

    precisions, recalls = calc_precision_recall(net, 
        loss_criterion, dataset, test_subset,
        batch_size=args.batch_size,
        cuda=cuda)
    if args.visdom_plot:
        import visdom
        viz = visdom.Visdom()
        viz.line(X=np.array(recalls), Y=np.array(precisions), opts=dict(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision"),env="main")
    print(list(zip(range(len(recalls)),recalls, precisions)))
    #TODO: allow loading
    torch.save({
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, "q3_checkpoint.pth.tar")

# python3 q3.py --cuda --visdom --epochs 400 --weight-decay 0
if __name__=="__main__":
    main()


