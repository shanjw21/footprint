import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False):
        super(Block,self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        if self.downsample:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        if self.downsample:
            out = self.pool(out)
        return out


class Transitaion(nn.Module):
    def __init__(self, in_channel):
        super(Transitaion,self).__init__()
        self.in_channel = in_channel
        self.bn = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        return out


class Identity(nn.Module):
    def __init__(self, in_channel, out_channel, pooling_size):
        super(Identity,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pooling_size = pooling_size
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pooling_size, stride=self.pooling_size))

    def forward(self, x):
        out = self.identity(x)
        return out


class ShoeNet(nn.Module):
    # in_channel: [1,32,64,128,256,512], out_channel: [32,64,128,256,512]
    def __init__(self):
        super(ShoeNet,self).__init__()
        self.trans = Transitaion(3)
        self.block_a = Block(3, 32)

        self.identity_atob = Identity(32, 32, 4)
        self.identity_atoc = Identity(32, 32, 8)
        self.identity_atod = Identity(32, 32, 16)
        self.identity_atoe = Identity(32, 32, 32)

        self.trans_a = Transitaion(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block_b = Block(32, 64, downsample=True)
        self.trans_b = Transitaion(96)

        self.identity_btoc = Identity(96, 32, 2)
        self.identity_btod = Identity(96, 32, 4)
        self.identity_btoe = Identity(96, 32, 8)

        self.block_c = Block(96, 128, downsample=True)
        self.trans_c = Transitaion(192)

        self.identity_ctod = Identity(192, 32, 2)
        self.identity_ctoe = Identity(192, 32, 4)

        self.block_d = Block(192, 256, downsample=True)
        self.trans_d = Transitaion(352)

        self.identity_dtoe = Identity(352, 32, 2)

        self.block_e = Block(352, 512, 2)
        self.trans_e = Transitaion(640)

        self.dropout1 = self.dropout1 = nn.Dropout(0.8)

        self.fc = nn.Sequential(nn.Linear(640 * 7 * 7, 384),
                                nn.Dropout(0.7),
                                nn.Linear(384, 384),
                                nn.Dropout(0.6),
                                nn.Linear(384, 384),
                                nn.Dropout(0.5),
                                nn.Linear(384, 1))

    def forward(self, x):
        out = self.trans(x)
        out = self.block_a(out)

        identity_atob = self.identity_atob(out)
        identity_atoc = self.identity_atoc(out)
        identity_atod = self.identity_atod(out)
        identity_atoe = self.identity_atoe(out)

        out = self.trans_a(out)
        out = self.pool(out)

        out = self.block_b(out)
        out = torch.cat([identity_atob, out], 1)
        out = self.trans_b(out)

        identity_btoc = self.identity_btoc(out)
        identity_btod = self.identity_btod(out)
        identity_btoe = self.identity_btoe(out)

        out = self.block_c(out)
        out = torch.cat([identity_atoc, identity_btoc, out], 1)
        out = self.trans_c(out)

        identity_ctod = self.identity_ctod(out)
        identity_ctoe = self.identity_ctoe(out)

        out = self.block_d(out)
        out = torch.cat([identity_atod, identity_btod, identity_ctod, out], 1)
        out = self.trans_d(out)

        identity_dtoe = self.identity_dtoe(out)

        out = self.block_e(out)
        out = torch.cat([identity_atoe, identity_btoe, identity_ctoe, identity_dtoe, out], 1)
        out = self.trans_e(out)

        out = self.dropout1(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    x = torch.rand([8,3,224,224])
    model = ShoeNet()
    print(model)
    out = model(x)
    print(out)