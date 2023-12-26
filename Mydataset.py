import torch
from torch.utils.data import Dataset
import numpy as np
import os
import time
from PIL import Image

class Mydataset(Dataset):
    def __init__(self, data_path:str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = np.load(self.data_path, allow_pickle=True, encoding="ASCII")

    def __getitem__(self, idx):
        image, label = self.data[idx][3], self.data[idx][2]
        img = Image.fromarray(image)
        label = torch.tensor(label).float()
        

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    cur_path = os.getcwd() # /data/shanjiawang/footprint
    data_path = os.path.join(cur_path,"dataset/X_train.npy")
    print(data_path)

    time1 = time.time()

    dataset = Mydataset(data_path)
    img, lable = dataset.__getitem__(1)
    print(img,lable)
    time2 = time.time()

    # print(type(img), img.size())
    print(type(lable),lable.size())
    print("load data costs {:.2f} seconds.".format(time2 - time1))