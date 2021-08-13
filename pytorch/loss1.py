# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import mmd
from sklearn.cluster import KMeans

def EntropyLoss(input_):
    mask = input_.ge(0.000001)  # mask is 0 if elem < 0.0000001,
    mask_out = torch.masked_select(input_, mask) # filter 0 in input_
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))


def Dann(features,ad_net, grl_layer, weights, use_gpu=True):
    loss = nn.BCELoss(weight=weights)
    # loss2 = nn.BCELoss(weight=weights,reduction="none")
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()
    if use_gpu:
        dc_target = dc_target.cuda()
    return loss(ad_out.view(-1), dc_target.view(-1)),ad_out.detach()


class sampler_cross_entropy(nn.Module):
    def __init__(self, weights):
        super(sampler_cross_entropy,self).__init__()
        self.weights = weights   # size of weights equal to batchsize
        if torch.sum(weights):
            self.active_ratio = len(weights) / torch.sum(weights)
        else:
            self.active_ratio = 0
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self, inputs, targets):
        self.weights = self.weights.expand_as(inputs)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        loss = (self.weights * (- targets) * log_probs).mean(0).sum() * self.active_ratio
        return loss



class class_crossentropy(nn.Module):
    def __init__(self, weights):
        super(class_crossentropy,self).__init__()
        self.weights = weights   # size of weights equal to num_class
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, inputs, targets):
        self.weights = self.weights.view(1,-1).expand_as(inputs)
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1).cuda()
        ratio = inputs.size(0) / torch.sum(self.weights * targets)
        loss = (self.weights * (- targets) * log_probs).mean(0).sum() * ratio
        return loss





def muiltlabelloss(features, target, multi_net,log_file,i):
    loss = nn.BCELoss()
    multi_feat = multi_net(features)

    # if (i +1) % 100 == 0:
    #     log_file.write(str(multi_feat.cpu().detach().numpy()[119:139,...].tolist())+'\n')
        # log_file.write(str(target.cpu().detach().numpy()[119:139, ...].tolist()) + '\n')
    return loss(multi_feat,target.cuda())



def NBCELoss(input_):
    mask = input_.ge(0.7)  # mask is 0 if elem < 0.7,
    mask_out = torch.masked_select(input_, mask) # filter 0 in input_
    if len(mask_out):
        entropy = -(torch.sum(mask_out * torch.log(mask_out))) / float(mask_out.size(0))
    else:
        entropy = 0
    mask2 = input_.ge(0.0000001) # mask is 0 if elem < 0.0000001
    mask3 = input_.le(0.1)  # mask is 0 if elem > 0.1
    mask3 = mask2 * mask3 # mask is 1 if 0 < elem < 0.1
    mask_out2 = torch.masked_select(input_, mask3)
    if len(mask_out2):
        entropy1 = -(torch.sum((1-mask_out2) * torch.log(1-mask_out2))) / float(mask_out2.size(0))
    else:
        entropy1 = 0
    return entropy + entropy1 #1/(number of negative label)* sum((1-mask_out2) * log(1-mask_out2))+1/(number of positive label)* sum((1-mask_out) * log(1-mask_out))

def NBCEWLoss(input_,od):
    input_ = input_.cuda()
    # print(input_.size())
    epsilon = 1e-6
    od = 1 - od
    # print(od)
    batch_size = od.size(1) // 2
    # print(batch_size,od.size(1))
    # od1 = Variable(torch.from_numpy(np.array([[1]] * batch_size)).float())
    od1 = torch.ones(1, batch_size).cuda()
    od1 = od1.squeeze(0)
    # print('od1', od1)
    od2 = od[0,batch_size: od.size(1)]
    # print("od2",od2)
    od = torch.cat((od1, od2), dim=0)
    # print('od',od)
    od = od.view(-1,1)
    # print(input_)
    input_ = od* input_
    mask = input_.ge(0.7)  # mask is 0 if elem >= 0.7,
    mask_hw = input_.le(0.2)  # mask is 0 if elem <= 0.2,
    # print(input_)
    mask_out = torch.masked_select(input_, mask)  #分类概率大的留下，当成伪标签
    mask_out1 = torch.masked_select(input_, mask_hw)  # 分类概率xiao的留下，当成伪标签
    # print(mask_out)
    # print(mask_out1)
    # print(mask_out,mask_out.size())
    # if len(mask_out) and not len(mask_out1):
    #     entropy = -(torch.sum(mask_out * torch.log(mask_out+epsilon))) / float(mask_out.size(0))
        # return entropy
    # elif len(mask_out1) and not len(mask_out):
    #     entropy1 = -(torch.sum((1-mask_out1) * torch.log(1-mask_out1+epsilon))) / float(mask_out1.size(0))
        # return entropy1
    # elif not len(mask_out1) and not len(mask_out):
    entropy = -(torch.sum(mask_out * torch.log(mask_out+epsilon))) / float(mask_out.size(0))
    entropy1 = -(torch.sum((1 - mask_out1) * torch.log(1 - mask_out1+epsilon))) / float(mask_out1.size(0))
    print(mask_out)
    print(mask_out1)
    return entropy + entropy1 #1/(number of negative label)* sum((1-mask_out2) * log(1-mask_out2))+1/(number of positive label)* sum((1-mask_out) * log(1-mask_out))

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN1(input_list, ad_net, weights, entropy=None, coeff=None, random_layer=None):
        softmax_output = input_list[1].detach()
        feature = input_list[0]
        if random_layer is None:
            op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
            ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        else:
            random_out = random_layer.forward([feature, softmax_output])
            ad_out = ad_net(random_out.view(-1, random_out.size(1)))
        batch_size = softmax_output.size(0) // 2
        dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
        if entropy is not None:
            entropy.register_hook(grl_hook(coeff))
            entropy = 1.0+torch.exp(-entropy)
            source_mask = torch.ones_like(entropy)
            source_mask[feature.size(0)//2:] = 0
            source_weight = entropy*source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[0:feature.size(0)//2] = 0
            target_weight = entropy*target_mask
            weight = source_weight / torch.sum(source_weight).detach().item() + \
                     target_weight / torch.sum(target_weight).detach().item()
            return torch.sum(weight.view(-1, 1)* weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item(), ad_out.detach()
        elif torch.sum(weights.view(-1, 1)):
            return nn.BCELoss(weight=weights.view(-1))(ad_out.view(-1), dc_target.view(-1)), ad_out.detach()
        else:
            return torch.tensor(0.0).cuda(), ad_out.detach()

def CDAN2(input_list, ad_net, od, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    od = od.float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        od = 1.0 + od
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = od*entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

if __name__=="__main__":
    import numpy as np
    print(np.empty((0,3,3)))
#     a= bin(int('94974F937D4C2433',16))
#     print(a)
#     b = bin(564897)
#     print(b)
#     print(a^b)

