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
import sys

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def get_monte_carlo_predictions(input_list,
                                forward_passes,
                                ad_net,
                                n_classes,
                                n_samples,random_layer=None):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes
    Parameters
    ----------
    inout_list : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    adnet : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """
    # ad_net.eval()
    print("yeah!")
    dropout_predictions = np.empty((0, n_samples, n_classes))
    for i in range(forward_passes):
        softmax_output = input_list[1].detach()
        feature = input_list[0]

        enable_dropout(ad_net)
        if random_layer is None:
            op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
            ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
        else:
            random_out = random_layer.forward([feature, softmax_output])
            ad_out = ad_net(random_out.view(-1, random_out.size(1)))
        ad_out = ad_out.cpu().detach().numpy()
        dropout_predictions = np.vstack((dropout_predictions, ad_out[np.newaxis,:,:]))
        # print(dropout_predictions)
    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)
    # Calculating variance across multiple MCD forward passes
    variance = np.var(dropout_predictions, axis=0)  # shape (n_samples, n_class
    # print(variance)
    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions * np.log(dropout_predictions + epsilon),axis=-1),axis=0)
    return variance



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

def get_cls_thresh(delta):  # delta: len(source) * 1
    kmeans = KMeans(n_clusters=2).fit(delta)
    if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
        thresh = min(delta[np.where(kmeans.labels_ == 1)])
    else:
        thresh = min(delta[np.where(kmeans.labels_ == 0)])
    return thresh[0]

# def classify_thresh(cls, labels): # len(source) * num_classes
def get_mc_var(dset_loaders,base_network, ad_net,random_layer, src_ones,tar_ones):
    with torch.no_grad():
        src_features_list = []
        tar_features_list = []
        src_softmax_out_list = []
        tar_softmax_out_list = []

        for p in iter(dset_loaders["source4test"]):
            features, outputs = base_network(p[0].cuda())
            softmax_output = nn.Softmax(dim=1)(outputs)
            src_features_list.append(features)
            src_softmax_out_list.append(softmax_output)
        for q in iter(dset_loaders["target4test"]):
            features, outputs = base_network(q[0].cuda())
            softmax_output = nn.Softmax(dim=1)(outputs)
            tar_features_list.append(features)
            tar_softmax_out_list.append(softmax_output)
        src_features = torch.cat(src_features_list, 0)
        src_softmax_out = torch.cat(src_softmax_out_list, 0)
        tar_features = torch.cat(tar_features_list, 0)
        tar_softmax_out = torch.cat(tar_softmax_out_list, 0)

        src_variance = get_monte_carlo_predictions([src_features, src_softmax_out], 10, ad_net, 1, 498, random_layer)
        tar_variance = get_monte_carlo_predictions([tar_features, tar_softmax_out], 10, ad_net, 1, 2817, random_layer)
        # thresh_all = get_cls_thresh(np.concatenate([src_variance, tar_variance]))
        src_transferbility = torch.tensor(src_variance)/torch.max(torch.tensor(src_variance))
        thresh_src = get_cls_thresh(src_transferbility)
        tar_transferbility = torch.tensor(tar_variance)/torch.max(torch.tensor(tar_variance))
        thresh_tar = get_cls_thresh(tar_transferbility)
        print("max_value",torch.max(torch.tensor(src_variance)),torch.max(torch.tensor(tar_variance)))
        print("thresh",thresh_src,thresh_tar)
        src_transferbility = torch.where(torch.tensor(src_transferbility).view(1, -1).cuda() > thresh_src.cuda(), src_ones,
                                         torch.zeros_like(src_ones))
        tar_transferbility = torch.where(torch.tensor(tar_transferbility).view(1, -1).cuda() > thresh_tar.cuda(), tar_ones,
                                         torch.zeros_like(tar_ones))
        print(src_transferbility)
        print("ahhh")
        print(tar_transferbility)
    return src_transferbility,tar_transferbility
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
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(),
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)
    dset_loaders["source4test"] = DataLoader(dsets["source"], batch_size=train_bs,
                                             shuffle=False, num_workers=4)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(),
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)
    dset_loaders["target4test"] = DataLoader(dsets["target"], batch_size=train_bs,
                                             shuffle=False, num_workers=4)

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
    len_sample_source = len(dsets["source"])
    len_sample_target = len(dsets["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    # alpha_i = 20

    cls_thresh = 0.0
    sharpen_classifer_critertion = nn.CrossEntropyLoss()
    src_ones = torch.ones(1, len_sample_source).cuda()
    tar_ones = torch.ones(1, len_sample_target).cuda()
    src_transferbility = torch.ones(1, len_sample_source).cuda()
    tar_transferbility = torch.ones(1, len_sample_target).cuda()

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
                # torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc1)
            config["out_file"].write(log_str + "\n")
            config["out_file"].flush()
            print(log_str)
        ###################sharpen part ###########################
        if (i+1) % config["update_iter"] == 0:
            with torch.no_grad():
                base_network.train(False)
                ad_net.train(False)
                source_fc8_out, source_labels = image_classification_predict(dset_loaders,
                                                              base_network,
                                                              test_10crop= False,
                                                              softmax_param=config["softmax_param"])
                one_hot_label = np.eye(class_num)[source_labels].astype(np.float32)
                samples_per_class = np.sum(one_hot_label, axis=0)
                delta_cls = np.sum(np.abs(one_hot_label - source_fc8_out), axis=0) / samples_per_class
                cls_weight = torch.tensor(delta_cls / np.mean(delta_cls)).cuda().view(-1).detach()
                sharpen_classifer_critertion = nn.CrossEntropyLoss(weight=cls_weight)


                ########### mc_dropout###############
                src_transferbility,tar_transferbility = get_mc_var(dset_loaders,
                                                                   base_network,
                                                                   ad_net,
                                                                   random_layer,
                                                                   src_ones,
                                                                   tar_ones)

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
        inputs_source, labels_source, index_source = iter_source.next()
        inputs_target, labels_target, index_target = iter_target.next()

        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        # print(sum(weight_ad))
        ones = torch.ones(1, 72).cuda()
        # zeros = torch.zeros(1, 72).cuda()
        # print(variance)
        src_trans = (src_transferbility.view(-1,1)[index_source]).view(1,-1)
        tar_trans = (tar_transferbility.view(-1,1)[index_target]).view(1,-1)
        all_transferbility = torch.cat((src_trans, tar_trans), dim=1)

        od = all_transferbility.cuda()
        # print("od", od)
        #####################################################
        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss1.CDAN2([features, softmax_out], ad_net, od, entropy, network.calc_coeff(i), random_layer)
            # transfer_loss0, _ = loss1.CDAN1([features, softmax_out], ad_net, ones, entropy, network.calc_coeff(i), random_layer)
            # transfer_loss1, _ = loss1.CDAN1([features, softmax_out], ad_net1, od, None, None, random_layer)
            # transfer_loss = transfer_loss0
        elif config['method'] == 'CDAN':
            transfer_loss0, _ = loss1.CDAN1([features, softmax_out], ad_net, ones, None, None, random_layer)
            # transfer_loss1, _ = loss1.CDAN1([features, softmax_out], ad_net1, od, None, None, random_layer)
            transfer_loss = transfer_loss0
            # transfer_loss = transfer_loss0
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        # entropy_loss = loss.EntropyLoss(nn.Softmax(dim=1)(outputs))
        # entropy_loss = loss1.NBCELoss(nn.Softmax(dim=1)(outputs))
        # print(outputs)

        # entropy_loss = loss1.EntropyLoss(nn.Softmax(dim=1)(outputs))
        sharpen_classifer_loss = sharpen_classifer_critertion(outputs.narrow(0, 0, inputs.size(0) // 2), labels_source)
        if i < 500:
            total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + sharpen_classifer_loss
        else:
            entropy_loss = loss1.NBCEWLoss(nn.Softmax(dim=1)(outputs),od)
            total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + sharpen_classifer_loss + 0.05 * entropy_loss
        total_loss.backward()
        optimizer.step()
        # if i % 10 == 0 and entropy_loss:
        #     print("iter: {:05d}, classifier_loss: {:.5f}".format(i, classifier_loss))
        #     print(
        #         i + 1,
        #         transfer_loss.cpu().item(),
        #         # transfer_loss1.cpu().item(),
        #         classifier_loss.cpu().item(),
        #         sharpen_classifer_loss.cpu().item(),
        #         entropy_loss.cpu().item(),
        #         total_loss.cpu().item())
        # elif i % 10 == 0 and not entropy_loss:
        print("iter: {:05d}, classifier_loss: {:.5f}".format(i, classifier_loss))
        print(
                i + 1,
                transfer_loss.cpu().item(),
                # transfer_loss1.cpu().item(),
                classifier_loss.cpu().item(),
                sharpen_classifer_loss.cpu().item(),
                total_loss.cpu().item())
        # tensor_writer.add_scalar('total_loss', total_loss, i)
        # tensor_writer.add_scalar('classifier_loss', classifier_loss, i)
        # tensor_writer.add_scalar('transfer_loss', transfer_loss, i)
    # torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='CDAN+E', choices=['CDAN', 'CDAN+E', 'DANN'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50',
                        choices=["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "VGG11", "VGG13",
                                 "VGG16", "VGG19", "VGG11BN", "VGG13BN", "VGG16BN", "VGG19BN", "AlexNet"])
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'image-clef', 'visda', 'office-home'],
                        help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='/home/tuo/jhu/PADA-master/pytorch/data/office/dslr_31_list.txt',
                        help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='/home/tuo/jhu/PADA-master/pytorch/data/office/amazon_31_list.txt',
                        help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--update_interval', type=int, default=500, help="interval of two continuous update phase")
    parser.add_argument('--snapshot_interval', type=int, default=500, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san',
                        help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--seed', type=float, default=444, help="interval of seed")
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
    config["loss"] = {"trade_off": 1.0,"seed": args.seed}

    loss_params = config["loss"]
    seed = loss_params["seed"]
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

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