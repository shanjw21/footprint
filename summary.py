#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary
from model.resnet import resnet34, resnet50
from model.resnet_cbam import resnet34_cbam, resnet50_cbam,resnet18_cbam
from model.convnext import convnext_tiny
from model.shoenet import ShoeNet

if __name__ == "__main__":
    input_shape     = [416, 416]
    num_classes     = 80
    net = resnet18_cbam

    summary(net, (3, input_shape[0], input_shape[1]))
    dummy_input = torch.randn(1,3,input_shape[0],input_shape[1])
    flops, params = profile(net,(dummy_input,),verbose=False)
    flops, params = clever_format([flops,params],"%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
