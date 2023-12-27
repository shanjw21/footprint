import argparse
import logging
import os
import sys
import torch
from tqdm import tqdm
import json
from torch import nn,optim
from torchvision import transforms
from Mydataset import Mydataset
from torch.utils.tensorboard import SummaryWriter
from utils import split_dataset,setup_seed,getLogger,clf_m,metric,plt_res
from model.resnet import resnet34, resnet50
from model.resnet_cbam import resnet34_cbam, resnet50_cbam,resnet18_cbam
from model.convnext import convnext_tiny
from model.shoenet import ShoeNet


func_dict = {
    "MSELoss":nn.MSELoss(),
    "CrossEntropyLoss":nn.CrossEntropyLoss(),
    "clf_m":clf_m
}

model_dict = {
    'resnet34':resnet34(),
    'resnet50':resnet50(),
    'resnet18_cbam':resnet18_cbam(),
    'resnet34_cbam':resnet34_cbam(),
    'resnet50_cbam':resnet50_cbam(),
    'convnext_tiny':convnext_tiny()
}


class Config:
    def __init__(self):
        self.experiment_name = "default"
        self.model_name = "resnet50"
        self.epochs = 100
        self.device = "cudu:0"
        self.batch_size = 32
        self.loss_fn = "clf_m"

    def get_path(self, path):
        return f"logs/{self.model_name}_{path}"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)


def eval_model(model, val_dataloader, device,loss_fn):
    model.eval()
    correct = 0
    eval_loss = 0
    eval_num_iter_per_epoch = len(val_dataloader)
    for batch_index, (images, targets) in enumerate(val_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        feature = model(images)

        loss = loss_fn(feature,targets)

        eval_loss += loss.item()
        correct += metric(targets,feature).sum().item()

    accuracy = (correct / len(val_dataloader.dataset))
    eval_loss /= eval_num_iter_per_epoch

    return eval_loss, accuracy

def train(args):
    setup_seed(1234)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("using {} device.".format(device))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    logger.info('Using {} dataloader workers every process'.format(nw))

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.0635,0.0635,0.0635], [0.1691,0.1691,0.1691])])
    
    data_path = os.path.join(os.path.dirname(os.getcwd()),"dataset/concate_testdata_inverse.npy")
    datasets = Mydataset(data_path,transform=data_transform)

    train_sampler, val_sampler = split_dataset(data_path)

    train_loader = torch.utils.data.DataLoader(datasets,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=nw,
                                               )
    
    eval_loader = torch.utils.data.DataLoader(datasets,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             num_workers=nw)
    
    if not model_dict.get(args.model) == None:
        model = model_dict[args.model]
        model.to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}. Available models are {list(model_dict.keys())}.")
    
    if not func_dict.get(args.loss_func) == None:
        loss_function = func_dict[args.loss_func]
    else:
        raise  ValueError(f"Invalid loss function: {args.loss_func}. Available loss functions are {list(func_dict.keys())}.")

    # 指定优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001, betas=(0.99,0.999),eps=1e-08,weight_decay=0.001)

    tb_writer = SummaryWriter(log_dir=f"./runs/{args.name}")

    train_num_iter_per_epoch = len(train_loader)
    logger.info(f"train_num_iter_per_epoch is {train_num_iter_per_epoch}")

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = 0
        train_correct = 0
        model.train()
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            features = model(images)
            train_correct += metric(features,targets).sum().item()

            loss = loss_function(features, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc = (train_correct / len(train_loader.dataset))
        train_loss /= train_num_iter_per_epoch

        val_loss, val_acc = eval_model(model,eval_loader,device,loss_function)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        eval_loss_list.append(val_loss)
        eval_acc_list.append(val_acc)
        learnning_rate_list.append(optimizer.param_groups[0]["lr"])
        
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        save_dir = f"weights/{args.name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if epoch % 5 == 0 or val_acc > best_acc:
            save_path = os.path.join(save_dir,f"{epoch:03d}_{val_acc}.pth")
            if val_acc > best_acc:
                best_acc = val_acc
                save_path = os.path.join(save_dir,f"{epoch:03d}_{val_acc}_best.pth")

            logger.info(f"Save model to {save_path}, Accuracy: {val_acc:.5f}")
            torch.save(model.state_dict(), save_path)

        logger.info(f"Epoch: {epoch} / {args.epochs}, Loss: {train_loss:.3f}, Accuracy: {val_acc:.5f}")
    logger.info(f"{args.epochs} finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--loss_func',type=str,default="clf_m")
    args = parser.parse_args()

    config = Config()
    config.experiment_name = args.name
    config.model_name = args.model
    config.epochs = args.epochs
    config.device = args.device
    config.batch_size = args.batch_size
    config.loss_fn = args.loss_func

    logger = getLogger(config.get_path("log.log"))
    logger.info(config)

    train_loss_list= []
    eval_loss_list = []
    train_acc_list = []
    eval_acc_list = []
    learnning_rate_list = []
    path = f"res_pic/{args.name}"

    train(args)
    plt_res(path,train_loss_list,train_acc_list,eval_loss_list,eval_acc_list,learnning_rate_list)