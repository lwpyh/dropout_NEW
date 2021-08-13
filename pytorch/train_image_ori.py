import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss, loss1
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import random
import pdb
import math
from sklearn.cluster import KMeans

def image_classification_predict(loader, model, test_10crop=True, gpu=True, softmax_param=1.0):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels.cuda()
            else:
                for j in range(10):
                    inputs[j] = inputs[j]
                labels = labels
            outputs = []
            for j in range(10):
                _, predict_out,_ = model(inputs[j])
                outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
            softmax_outputs = sum(outputs)
            if start_test:
                all_softmax_output = softmax_outputs.data.cpu().float()
                start_test = False
            else:
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    else:
        iter_val = iter(loader["source"])
        for i in range(len(loader['source'])):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = inputs.cuda()
            else:
                inputs = inputs
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
            if start_test:
                all_labels = labels
                all_softmax_output = softmax_outputs.data.cpu().float()
                start_test = False
            else:
                all_labels = torch.cat((all_labels, labels), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
    return np.array(all_softmax_output), np.array(all_labels)

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    acc = []
    # print(set(all_label))
    # for i in set(all_label):
    #     # print(torch.squeeze(predict[all_label==i]).float())
    #     acc.append(torch.sum(torch.squeeze(predict[all_label == i]).float() == all_label[all_label == i]) / float(
    #         all_label[all_label == i].size()[0]))
    accuracy = float(torch.sum(torch.squeeze(predict).float() == all_label)) / float(all_label.size()[0])
    return accuracy, acc

def get_thresh(loader, model, ad_net, i):    ############################new#######################
    thresh_range, thresh_range_1 = [], []
    for p in iter(loader["source"]):
        features, outputs = model(p[0].cuda())
        softmax_out = nn.Softmax(dim=1)(outputs).detach()
        input_list = [features, softmax_out]
        softmax_output = input_list[1].detach()
        feature = input_list[0]
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1))).detach()
        thresh_range.extend(ad_out.cpu().numpy())
    thresh_range1, thresh_range2, thresh_range3 = [], [], []
    for m in thresh_range:
        if np.abs(m - 0.5) < 0.05:
            if m > 0.5:
                thresh_range1.append(2*m-1)
            else:
                thresh_range1.append(1-2*m)
        elif np.abs(m - 0.5) < 0.2:
            if m > 0.5:
                thresh_range2.append(2*m-1)
            else:
                thresh_range2.append(1-2*m)
        else:
            if m > 0.5:
                thresh_range3.append(2*m-1)
            else:
                thresh_range3.append(1-2*m)
    for q in iter(loader["target"]):
        features, outputs = model(q[0].cuda())
        softmax_out = nn.Softmax(dim=1)(outputs).detach()
        input_list = [features, softmax_out]
        softmax_output = input_list[1].detach()
        feature = input_list[0]
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1))).detach()
        thresh_range_1.extend(ad_out.cpu().numpy())  #get all samples' domain output
    thresh_range = np.array(thresh_range).reshape(-1, 1)
    print(ad_out)

    # thresh_range_t = np.abs(np.array(thresh_range) - 0.5).reshape(-1, 1)
    for m in thresh_range_1:
        if np.abs(m - 0.5) < 0.05:
            if m < 0.5:
                thresh_range1.append(1-2*m)
            else:
                thresh_range1.append(2*m-1)
        elif np.abs(m - 0.5) < 0.2:
            if m < 0.5:
                thresh_range2.append(1-2*m)
            else:
                thresh_range2.append(2*m-1)
        else:
            if m < 0.5:
                thresh_range3.append(1-2*m)
            else:
                thresh_range3.append(2*m-1)
    # print(thresh_range1, thresh_range2)
    if len(thresh_range1):
        thresh1 = 1 / len(thresh_range1) * np.sum(thresh_range1)
    else:
        thresh1 = 0
    if len(thresh_range2):
        thresh2 = 1 / len(thresh_range2) * np.sum(thresh_range2)
    else:
        thresh2 = 0
    if len(thresh_range3):
        thresh3 = 1 / len(thresh_range3) * np.sum(thresh_range3)
    else:
        thresh3 = 0
    print(thresh1, thresh2, thresh3)
    return thresh1,thresh2, thresh3


def get_cls_thresh(delta):  # delta: len(source) * 1
    kmeans = KMeans(n_clusters=2).fit(delta)
    if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
        thresh = min(delta[np.where(kmeans.labels_ == 1)])
    else:
        thresh = min(delta[np.where(kmeans.labels_ == 0)])
    return thresh[0]

# def classify_thresh(cls, labels): # len(source) * num_classes


def train(config):
    ## set pre-process
    tensor_writer = SummaryWriter(config["tensorboard_path"])
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=True, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
        ad_net1 = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
        ad_net1 = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    ad_net1 = ad_net1.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters() + ad_net1.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        ad_net1 = nn.DataParallel(ad_net1, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    # alpha_i = 20
    thresh1 = thresh2 = thresh3 = 0.0
    cls_thresh = 0.0
    sharpen_classifer_critertion = nn.CrossEntropyLoss()

    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                                                 base_network, test_10crop=prep_config["test_10crop"])
            temp_acc1 = temp_acc[0]
            temp_model = nn.Sequential(base_network)
            if temp_acc1 > best_acc:
                best_acc = temp_acc1
                best_model = temp_model
                torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc1)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
        ###################sharpen part ###########################
        if (i) % config["update_iter"] == 0:
            base_network.train(False)
            ad_net.train(False)
            thresh1, thresh2, thresh3 = get_thresh(dset_loaders, base_network, ad_net, i)
            source_fc8_out, source_labels = image_classification_predict(dset_loaders,
                                                          base_network,
                                                          test_10crop= False,
                                                          softmax_param=config["softmax_param"])
            one_hot_label = np.eye(class_num)[source_labels].astype(np.float32)
            samples_per_class = np.sum(one_hot_label, axis=0)
            delta_cls = np.sum(np.abs(one_hot_label - source_fc8_out), axis=0) / samples_per_class
            cls_weight = torch.tensor(delta_cls / np.mean(delta_cls)).cuda().view(-1).detach()
            sharpen_classifer_critertion = nn.CrossEntropyLoss(weight=cls_weight)

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        ad_net1.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        # print(sum(weight_ad))
        ones = torch.ones(1, 72).cuda()
        list_t1 = torch.ones(1, 72).cuda() * thresh1
        list_t2 = torch.ones(1, 72).cuda() * thresh2
        list_t3 = torch.ones(1, 72).cuda() * thresh3
        # list_t1, list_t2, list_t3 = [], [], []
        # list_t1_w, list_t2_w, list_t3_w = [], [], []
        # for p in range(72):
        #     list_t1.append(thresh1)
        #     list_t1_w.append(1 - thresh1)
        # list_t1 = np.array(list_t1)
        # list_t1 = torch.from_numpy(list_t1).float().cuda()  # hard transferability
        # list_t1_w = np.array(list_t1_w)
        # list_t1_w = torch.from_numpy(list_t1_w).float().cuda()  # easy transferability
        # for p in range(72):
        #     list_t2.append(thresh2)
        #     list_t2_w.append(1 - thresh2)
        # list_t2 = np.array(list_t2)
        # list_t2 = torch.from_numpy(list_t2).float().cuda()  # hard transferability
        # list_t2_w = np.array(list_t2_w)
        # list_t2_w = torch.from_numpy(list_t2_w).float().cuda()  # esay transferability
        # for p in range(72):
        #     list_t3.append(thresh3)
        #     list_t3_w.append(1 - thresh3)
        # list_t3 = np.array(list_t3)
        # list_t3 = torch.from_numpy(list_t3).float().cuda()  # hard transferability
        # list_t3_w = np.array(list_t3_w)
        # list_t3_w = torch.from_numpy(list_t3_w).float().cuda()  # esay transferability
        #####################################################
        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss0, od = loss1.CDAN1([features, softmax_out], ad_net, ones, entropy, network.calc_coeff(i), random_layer)
            od1 = torch.where(torch.abs(od.view(1, -1) - 0.5) < 0.05, list_t1, list_t2)  # new
            od2 = torch.where(torch.abs(od.view(1, -1) - 0.5) < 0.2, od1, list_t3)  # new
            # print(od, od1,od2)
            transfer_loss1, od3 = loss1.CDAN1([features, softmax_out], ad_net1, od2, None, None, random_layer)
            transfer_loss = transfer_loss0 + 2 * transfer_loss1
        elif config['method'] == 'CDAN':
            transfer_loss0, od = loss1.CDAN1([features, softmax_out], ad_net, ones, None, None, random_layer)
            od1 = torch.where(torch.abs(od.view(1, -1) - 0.5) < 0.05, list_t1, list_t2)  # new
            # od2 = torch.where(torch.abs(od.view(1, -1) - 0.5) < 0.2, od1, list_t3)  # new
            # print(od,od1,od2)
            transfer_loss1, od3 = loss1.CDAN1([features, softmax_out], ad_net1, od2, None, None, random_layer)
            transfer_loss = transfer_loss0 + 2 * transfer_loss1
            # transfer_loss = transfer_loss0
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        # entropy_loss = loss1.NBCELoss(nn.Softmax(dim=1)(outputs))
        entropy_loss = loss1.NBCEWLoss(nn.Softmax(dim=1)(outputs),od2)
        sharpen_classifer_loss = sharpen_classifer_critertion(outputs.narrow(0, 0, inputs.size(0) // 2), labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + sharpen_classifer_loss + 1.4 * entropy_loss
        total_loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("iter: {:05d}, classifier_loss: {:.5f}".format(i, classifier_loss))
            print(
                i + 1,
                transfer_loss0.cpu().item(),
                transfer_loss1.cpu().item(),
                classifier_loss.cpu().item(),
                sharpen_classifer_loss.cpu().item(),
                entropy_loss.cpu().item(),
                total_loss.cpu().item())
        tensor_writer.add_scalar('total_loss', total_loss, i)
        tensor_writer.add_scalar('classifier_loss', classifier_loss, i)
        tensor_writer.add_scalar('transfer_loss', transfer_loss, i)
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/amazon_31_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/dslr_31_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--update_interval', type=int, default=500, help="interval of two continuous update phase")
    parser.add_argument('--snapshot_interval', type=int, default=500, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--seed', type=float, default=999, help="interval of seed")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    print(os.getcwd())

    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["softmax_param"] = 1.0
    config["num_iterations"] = 16004
    config["test_interval"] = args.test_interval
    config["update_iter"] = args.update_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = "snapshot/" + args.output_dir
    model_name = args.method + '/' + osp.basename(args.s_dset_path)[0].upper() + '2' + osp.basename(args.t_dset_path)[
        0].upper()
    config["tensorboard_path"] = "vis/" + model_name
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p ' + config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "a")
    if not osp.exists(config["tensorboard_path"]):
        os.system('mkdir -p ' + config["tensorboard_path"])

    config["prep"] = {"test_10crop": True, 'params': {"resize_size": 256, "crop_size": 224, 'alexnet': False}}
    config["loss"] = {"trade_off": 1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name": network.AlexNetFc, \
                             "params": {"use_bottleneck": True, "bottleneck_dim": 256, "new_cls": True}}
    elif "ResNet" in args.net:
        config["network"] = {"name": network.ResNetFc, \
                             "params": {"resnet_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    elif "VGG" in args.net:
        config["network"] = {"name": network.VGGFc, \
                             "params": {"vgg_name": args.net, "use_bottleneck": True, "bottleneck_dim": 256,
                                        "new_cls": True}}
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": 0.9, \
                                                               "weight_decay": 0.0005, "nesterov": True},
                           "lr_type": "inv", \
                           "lr_param": {"lr": args.lr, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36}, \
                      "target": {"list_path": args.t_dset_path, "batch_size": 36}, \
                      "test": {"list_path": args.t_dset_path, "batch_size": 4}}

    if config["dataset"] == "office":
        if ("amazon" in args.s_dset_path and "webcam" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("webcam" in args.s_dset_path and "amazon" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "amazon" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        elif ("amazon" in args.s_dset_path and "dslr" in args.t_dset_path) or \
                ("dslr" in args.s_dset_path and "webcam" in args.t_dset_path):
            config["optimizer"]["lr_param"]["lr"] = 0.0003  # optimal parameters
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001  # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)
