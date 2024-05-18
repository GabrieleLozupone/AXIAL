# -----------------------------------------------------------------------------
# This file is part of the project "Joint Learning Framework of Cross-Modal Synthesis
# and Diagnosis for Alzheimer's Disease by Mining Underlying Shared Modality Information".
# Original repository: https://github.com/thibault-wch/Joint-Learning-for-Alzheimer-disease.git
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


class AwareNet(nn.Module):
    def __init__(self, num_classes, return_attention_weights=False):
        super(AwareNet, self).__init__()
        # [1] base feature extractor
        self.basic_module = TimeDistributed(Basic_Fex())
        # [2] slice-aware module
        self.att_module = Attention()
        # [3] Fusion feature extractor
        self.fusion_module = Fusion_Fex(num_classes=num_classes)
        self.return_attention_weights = return_attention_weights

    def forward(self, x):
        x = self.basic_module(x, init=True)
        x, slice_attn, local_attn, alpha = self.att_module(x)
        x = self.fusion_module(x)
        x = torch.mean(x, axis=1)
        Y_hat = torch.softmax(x, dim=1).argmax(dim=1)
        if self.return_attention_weights:
            return x, Y_hat, slice_attn, local_attn, alpha
        return x


# [1] Basic feature extractor
class Basic_Fex(nn.Module):
    def __init__(self, expansion=4):
        super(Basic_Fex, self).__init__()
        self.expansion = expansion
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# [2] Slice-aware module
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.loc_att = TimeDistributed(
            nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1), nn.LeakyReLU(0.2, inplace=True)))
        self.k = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False)
        self.q = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.alpha = 32.0

    def forward(self, x):
        ori_x = x
        local_attention = self.loc_att(x)
        x = local_attention.squeeze(2)
        b, c, h, w = x.shape
        k = self.k(x).reshape(b, c, -1)
        q = self.q(x).reshape(b, 1, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)).squeeze(1)
        attn *= self.alpha
        slice_attention = attn.softmax(dim=-1)
        attn = slice_attention.reshape(b, 256, 1, 1, 1).expand(b, 256, 128, 32, 32)
        out = attn * ori_x
        return out + ori_x, slice_attention, local_attention, self.alpha


# [3] Fusion feature extractor
class Fusion_Fex(nn.Module):
    def __init__(self, num_classes=2, expansion=4):
        super(Fusion_Fex, self).__init__()
        self.expansion = expansion
        self.in_channels = 128
        # shared 3D->2D
        self.layer1 = TimeDistributed(self._make_layer(BasicBlock, 256, 2, 2))
        self.layer2 = TimeDistributed(self._make_layer(BasicBlock, 512, 2, 2))
        self.global_avg = TimeDistributed(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = TimeDistributed(
            nn.Sequential(*[nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, num_classes)]))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    # slice shift module
    def TemporalShiftModule(self, x, fold_div):
        # shape of x: [N, T, C, H, W]
        c = x.size(1)
        out = torch.zeros_like(x)
        fold = c // fold_div
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out

    def forward(self, x):
        # slice shift 1
        x = self.TemporalShiftModule(x, 8)
        x = self.layer1(x)
        # slice shift 2
        x = self.TemporalShiftModule(x, 8)
        x = self.layer2(x)
        x = self.global_avg(x)
        x = self.classifier(x, fcmodule=True)
        return x


############################################### Basic network components ###############################################
# [C.1] Time distribute 3D->2D
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x, fcmodule=False, init=False):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        if init:
            x = x.unsqueeze(2)
        if fcmodule:
            batch_size, time_steps, C, _, _ = x.size()
            c_in = x.reshape(batch_size, time_steps, C)
            c_out = self.module(c_in)
        else:
            batch_size, time_steps, C, H, W = x.size()
            c_in = x.view(batch_size * time_steps, C, H, W)
            c_out = self.module(c_in)
            _, c, h, w = c_out.size()
            c_out = c_out.reshape(batch_size, time_steps, c, h, w)
        return c_out


#  [C.2] Basic resnet block
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
