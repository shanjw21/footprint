import os
import json
from utils import getLogger
import argparse
import torch

def get_path(path):
    model = "resnet50"
    return f"logs/{model}_{path}"

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
    

def clf_m(predictions, labels, k=5.0):
    diff = torch.mean(torch.abs(predictions - labels))
    lossout = torch.mean(torch.where(diff <= k, (0.000001 * diff), (diff ** 3) + 0.1))
    return (lossout)


if __name__ == '__main__':
    # config = Config()
    # logger = getLogger(config.get_path(path="log.log"))
    # logger.info(config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss_func',default=clf_m,action='store_false')
    args = parser.parse_args()

    loss_fn = args.loss_func
    
    print(loss_fn)