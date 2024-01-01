import argparse
import logging
import os
import torch
import torch.nn as nn
import json
from torchvision import transforms
from Mydataset import Mydataset
from utils import getLogger,clf_m,metric
from model.resnet import resnet34, resnet50
from model.resnet_cbam import resnet34_cbam, resnet50_cbam,resnet18_cbam
from train import func_dict,model_dict


class Config:
    def __init__(self):
        self.experiment_name = "default"
        self.model_name = "resnet50"
        self.device = "cudu:0"
        self.weights_path = "./weights"

    def get_path(self, path):
        return f"logs/prediction/{self.model_name}_{path}"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

def test(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("using {} device.".format(device))

    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.0635,0.0635,0.0635], [0.1691,0.1691,0.1691])])
    
    data_path = os.path.join(os.path.dirname(os.getcwd()),"dataset/test_dataset.npy")
    test_datasets = Mydataset(data_path,transform=data_transform)

    batch_size = 32
    test_loader = torch.utils.data.DataLoader(test_datasets,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=8)
    if not model_dict.get(args.model) == None:
        model = model_dict[args.model]
        model.to(device)
    else:
        raise ValueError(f"Invalid model: {args.model}. Available models are {list(model_dict.keys())}.")
    
    weights_path = f"weights/{args.name}/{args.weights_path}.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    correct = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            feature = model(images)
            correct += metric(targets,feature).sum().item()
        accuracy = (correct / len(test_loader.dataset))
    logger.info('test_accuracy: %.3f ' % (accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--model",type=str,help="the name of model to prediction")
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--weights_path',type=str,help="specifiy the path to best_weights")

    args = parser.parse_args()
    config = Config()
    config.experiment_name = args.name
    config.model_name = args.model
    config.device = args.device
    config.weights_path = args.weights_path
    logger = getLogger(config.get_path("log.log"))
    logger.info(config)
    test(args)