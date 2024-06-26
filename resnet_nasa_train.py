# 2022.06.28-Changed for building CMT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# Modified from Fackbook, Deit
# jianyuan.guo@huawei.com
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from utils.tools import *
from model.crossformer import CrossFormer
from torch.autograd import Variable
import torch
import torch.optim as optim
import time
import timm
from loguru import logger

from apex import amp

torch.multiprocessing.set_sharing_strategy('file_system')
from relative_similarity import *
from centroids_generator import *
import torch.nn.functional as F
from loss.loss import RelaHashLoss

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
from timm.scheduler import create_scheduler, CosineLRScheduler
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler
from loguru import logger
import CrossFormer_utils as utils

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

import warnings

warnings.filterwarnings("ignore")  # Del ImageNet Warnings
import os

# from cmt_args import get_args_parser
# from swiftformer_args import get_args_parser
from crossFormer_args import get_config as get_configs
from crossFormer_args import parse_option
from loss.hypbird import margin_contrastive
from networks import ResNet50_
import loss.nasaloss as nasa


def build_model(config, args, bit):
    model = ResNet50_(bit)
    return model


def get_config():
    config = {
        "alpha": 0.1,

        'info': "[ResNet]",
        'loss_type': "Rela",
        "step_continuation": 20,
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        # "datasets": "mirflickr",
        "datasets": "cifar10",
        # "datasets":'nuswide_21',
        # "datasets":'coco',
        "Label_dim": 10,
        "epoch": 700,
        "test_map": 0,
        "save_path": "save/HashNet",
        "device": torch.device("cuda:0"),
        'test_device': torch.device("cuda:1"),
        "bit_list": [16, 32, 64, 128],
        "img_size": 224,
        "patch_size": 4,
        "in_chans": 3,
        "num_work": 10,
        "model_type": "ResNet",
        "top_img": 100,

        # RelaHash
        "Beta": 8,
        'm': 0.7,

    }
    config = config_dataset(config)
    return config


def train_val(config, bit, args=None, configs=None):
    device = torch.device(config['device'])

    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.manual_seed(3407)

    cudnn.benchmark = True

    print(f"Creating model: {args.cfg}")

    model = build_model(configs, args, bit)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    input_size = [1, 3, 224, 224]
    input = torch.randn(input_size).cuda()
    from torchprofile import profile_macs
    if 'asmlp' not in args.cfg:
        macs = profile_macs(model.eval(), input)
        print('model flops:', macs, 'input_size:', input_size)
        model.train()

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    # linear_scaled_lr = args.lr * args.batch_size  / 512.0
    args.lr = linear_scaled_lr
    print('learning rate: ', args.lr)
    optimizer = create_optimizer(args, model)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    loss_scaler = ApexScaler()
    print('Using NVIDIA APEX AMP. Training in mixed precision.')

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    model.to(config["device"])

    project = nasa.ProjectLayer(bit).to(device)
    nasa_loss = nasa.NASA_loss(device=device)
    triplet = nasa.DTSHLoss()

    optimizer_project = torch.optim.Adam(project.parameters(),lr=1e-4)
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    Best_mAP = 0

    for epoch in range(config["epoch"]):
        # model.module.update_temperature()

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        logger.info("%s[%2d/%2d][%s] bit:%d, datasets:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["datasets"]), end="")

        model.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                u = model(image)
                u = F.normalize(u)
                loss1 = nasa_loss(u,project(u))
                loss2 = triplet(u,label)
                train_loss += loss1.item() + loss2.item()
            optimizer.zero_grad()
            optimizer_project.zero_grad()

            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss1 + loss2, optimizer, clip_grad=None,
                        parameters=model.parameters(), create_graph=is_second_order)
            # loss_scaler(loss2, optimizer, clip_grad=None,
            #         parameters=model.parameters(), create_graph=is_second_order)
            optimizer_project.step()

        train_loss = train_loss / len(train_loader)

        # /home/wbt/conda_env_with_new_amp/conda_env/anaconda3/bin/python3.9 CMT_train.py --output_dir './' --model cmt_s --batch-size 32 --apex-amp --input-size 224 --weight-decay 0.05 --drop-path 0.1 --epochs 300 --test_freq 100 --test_epoch 260 --warmup-lr 1e-7 --warmup-epochs 20
        logger.info("\b\b\b\b\b\b\b train_loss:%.4f" % (train_loss))
        if (epoch + 1) > 300:
            Best_mAP, index_img = validate(config, Best_mAP, test_loader, dataset_loader, model, bit, epoch, 10)
            model.to(config["device"])


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(' traini ng and evaluation script', parents=[get_config()])
    # args = parser.parse_args()
    args, configs = parse_option()
    config = get_config()
    # 建立日志文件（Create log file）
    logger.add('logs/{time}' + config["info"] + '_' + config["datasets"] + ' alpha ' + str(config["alpha"]) + '.log',
               rotation='50 MB', level='DEBUG')

    logger.info(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/alexnet/HashNet_{config['datasets']}_{bit}.json"
        train_val(config, bit, args, configs)
