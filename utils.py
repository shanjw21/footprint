import logging
import os
import random
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from Mydataset import Mydataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def clf_m(predictions, labels, k=5.0):
    diff = torch.mean(torch.abs(predictions - labels))
    lossout = torch.mean(torch.where(diff <= k, (0.000001 * diff), (diff ** 3) + 0.1))
    return (lossout)


def metric(predictions, labels, k=5.0):
    diff = torch.abs(predictions - labels)
    return diff <= k


def train_one_epoch(model, optimizer, data_loader, device, epoch, loss_function):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        accu_num += metric(pred, labels.to(device)).sum().item()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.item()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            logging.info('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, loss_function):
    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        accu_num += metric(pred, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def model_onnx(model, model_name: str):
    """
    visualize the network architecture through onnx
    Args:
        model: the network. eg resnet18
        model_name: the name of network.
    """
    print(f"make {model_name}'s onnx file")
    x = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(model, x, model_name)
    print("finished!")



def setup_seed(seed):
    """
    set random seeds
    Args:
        seed: seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def split_dataset(data_path,split_proption=0.2,shuffle_or_not=True):
    """
    split the datasets to  train_datasets and val_datasets
    Args:
        data_path: the path to datasets
        split_proption: the proption of train and val
        shuffle_or_not:
    """    
    data_npy = np.load(data_path,allow_pickle=True)

    # set random seed
    random_seed = 1234

    # create data indices for trainning and testing splits
    dataset_size = len(data_npy)
    indices = list(range(dataset_size))

    # count out split size
    split = int(np.floor(split_proption * dataset_size))

    # split the train_indices and val_indices
    if shuffle_or_not:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:],indices[:split]
    print(f"the length of train_datasets is {len(train_indices)} and the length of val_datasets is {len(val_indices)}")
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def getLogger(path):
    """
    making a logger object to records.
    Args:
        path: the way log_file locates
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rf_handler = logging.FileHandler(path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s')
    rf_handler.setFormatter(formatter)
    logger.addHandler(rf_handler)

    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger


def plt_res(path:str,train_losses:list, train_accuracys:list, eval_losses:list, eval_accuracys:list,learning_rates:list)->None:
    """
    plots the result of every experiment.
    Args:
        path: the way to store the result pictures location.
        train_losses: the list contains all train_loss for this experiment.
        train_accuracys: the list contains all  train_accuracys for this experiment.
        eval_losses: the list contains all  eval_losses for this experiment.
        eval_accuracys:  the list contains all eval_accuracys for this experiment.
    """

    epochs = range(1,len(train_losses) + 1)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, eval_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracys, 'bo-', label='Training Accuracy')
    plt.plot(epochs, eval_accuracys, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, learning_rates, 'bo-', label='Learning Rate')
    plt.title('Learning Rate Curve')
    plt.xlabel('Epochs')
    plt.ylabel('learning_rates')
    plt.legend()

    if not os.path.exists(path):
        os.makedirs(path)
    filename = "training_plot.png"
    save_path = os.path.join(path,filename)
    plt.tight_layout()
    plt.savefig(save_path)