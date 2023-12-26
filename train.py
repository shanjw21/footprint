import argparse
import logging
import os
import json
import torch
from torch import nn,optim
from torchvision import transforms
from Mydataset import Mydataset
from torch.utils.tensorboard import SummaryWriter
from utils import train_one_epoch,evaluate,split_dataset,setup_seed,getLogger
from model.resnet import resnet34, resnet50
from model.resnet_cbam import resnet34_cbam, resnet50_cbam
from model.convnext import convnext_tiny
from model.shoenet import ShoeNet
from utils import clf_m


class Config:
    def __init__(self):
        self.model_name = "resnet50"
        self.epochs = 100
        self.batch_size = 32
        self.device = "cudu:0"
        self.loss_fn = "clf_m"

    def get_path(self, path):
        return f"logs/{self.model_name}_{path}"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)



def main(args):
    log_path = os.path.join(os.getcwd(),f'log/{args.model}_{args.batch_size}_{args.loss_func}.log')
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO,filename=log_path)
    logging.info(f'the current log path is :{log_path}')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.0635,0.0635,0.0635], [0.1691,0.1691,0.1691])]),
    
    data_path = os.path.join(os.path.dirname(os.getcwd()),"concate_traindata_inverse.npy")
    datasets = Mydataset(data_path,transform=data_transform)

    train_sampler, val_sampler = split_dataset(data_path)

    train_loader = torch.utils.data.DataLoader(datasets,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=nw,
                                               )
    
    val_loader = torch.utils.data.DataLoader(datasets,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             num_workers=nw)
    
    if  'resnet50' == args.model:
        model = resnet50().to(device)
    elif 'resnet34' == args.model:
        model = resnet34().to(device)
    elif 'resnet34_cbam' == args.model:
        model = resnet34_cbam().to(device)
    elif 'resnet50_cbam' == args.model:
        model = resnet50_cbam().to(device)
    elif 'convnext_tiny' == args.model:
        model = convnext_tiny(1).to(device)
    elif 'shoenet' == args.model:
        model = ShoeNet().to(device)
    else:
        raise ModuleNotFoundError
    
    loss_function=args.loss_func

    # 指定优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.001, betas=(0.99,0.999),eps=1e-08,weight_decay=0.001)

    tb_writer = SummaryWriter(log_dir=f"./runs/{args.model}_{loss_function}_{args.batch_size}")

    best_acc = 0.
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                loss_function=loss_function)
        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     loss_function=loss_function)
        
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./weights/best_{args.model}_{loss_function}_{args.batch_size}.pth")

        torch.save(model.state_dict(), f"./weights/latest_{args.model}_{loss_function}_{args.batch_size}.pth")
    
    logging.info(f"{args.epochs} finished!")
    print(f"{args.epochs} finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--loss_func',default=clf_m,action='store_false')
    args = parser.parse_args()

    config = Config()
    config.model_name = args.model
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.device = args.device


    logger = getLogger()

    main(args)