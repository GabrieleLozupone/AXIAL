# -----------------------------------------------------------------------------
# This file is part of the project "Joint Learning Framework of Cross-Modal Synthesis
# and Diagnosis for Alzheimer's Disease by Mining Underlying Shared Modality Information".
# Original repository: https://github.com/thibault-wch/Joint-Learning-for-Alzheimer-disease.git
# -----------------------------------------------------------------------------

from torch.nn import init
from torch.optim import lr_scheduler
from .awarenet import AwareNet
import torch


###############################################################################
# Helper Functions
###############################################################################
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = min(1.0, opt.lr_num ** (epoch - opt.niter))
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.2)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
    return net


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=''):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net)
        net.cuda()
    return init_weights(net, init_type, gain=init_gain)


def define_Cls(class_num=2, init_type='normal', init_gain=0.02, gpu_ids=[], joint=False):
    net = AwareNet(num_classes=class_num)
    if joint:
        return init_weights(net, init_type, gain=init_gain)
    else:
        return init_net(net, init_type, init_gain, gpu_ids)
