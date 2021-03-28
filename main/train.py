import __init_lib_path
from config_guard import cfg, update_config_from_yaml
import dataset
import backbone
import classifier
import loss
import utils

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description = "Roger's Deep Learning Playground")
    parser.add_argument('--cfg', help = "specify particular yaml configuration to use", required = True,
        default = "configs/mnist_torch_official.taml", type = str)
    args = parser.parse_args()

    return args

def train(cfg, model, post_processor, criterion, device, train_loader, optimizer, epoch):
    model.train()
    post_processor.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        feature = model(data)
        output = post_processor(feature)
        loss = criterion(output, target)
        optimizer.zero_grad() # reset gradient
        loss.backward()
        optimizer.step()
        if cfg.task == "classification":
            pred = output.argmax(dim = 1, keepdim = True)
            correct_prediction = pred.eq(target.view_as(pred)).sum().item()
            batch_acc = correct_prediction / data.shape[0]
            if batch_idx % cfg.TRAIN.log_interval == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Acc: {5:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), batch_acc))
        elif cfg.task == "semantic_segmentation":
            pred_map = output.max(dim = 1)[1]
            batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
            if batch_idx % cfg.TRAIN.log_interval == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tBatch Pixel Acc: {5:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), batch_acc))
        else:
            raise NotImplementedError


def test(cfg, model, post_processor, criterion, device, test_loader):
    model.eval()
    post_processor.eval()
    test_loss = 0
    correct = 0
    # TODO: use a more consistent evaluation interface
    pixel_acc_list = []
    iou_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            feature = model(data)
            output = post_processor(feature)
            test_loss += criterion(output, target).item()  # sum up batch loss
            if cfg.task == "classification":
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            elif cfg.task == "semantic_segmentation":
                pred_map = output.max(dim = 1)[1]
                batch_acc, _ = utils.compute_pixel_acc(pred_map, target, fg_only=cfg.METRIC.SEGMENTATION.fg_only)
                pixel_acc_list.append(float(batch_acc))
                for i in range(pred_map.shape[0]):
                    iou = utils.compute_iou(
                        np.array(pred_map[i].cpu()),
                        np.array(target[i].cpu(), dtype=np.int64),
                        cfg.num_classes,
                        fg_only=cfg.METRIC.SEGMENTATION.fg_only
                    )
                    iou_list.append(float(iou))
            else:
                raise NotImplementedError

    test_loss /= len(test_loader.dataset)

    if cfg.task == "classification":
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    elif cfg.task == "semantic_segmentation":
        print('\nTest set: Average loss: {:.4f}, Mean Pixel Accuracy: {:.4f}, Mean IoU {:.4f}\n'.format(
            test_loss, np.mean(pixel_acc_list), np.mean(iou_list)))
    else:
        raise NotImplementedError


def main():
    # --------------------------
    # | Initial set up
    # |  1. Parse arg and yaml
    # |  2. Set device
    # |  3. Set seed
    # --------------------------
    args = parse_args()
    update_config_from_yaml(cfg, args)

    use_cuda = not cfg.SYSTEM.use_cpu
    device_str = "cuda" if use_cuda else "cpu"
    device = torch.device(device_str)

    kwargs = {'num_workers': cfg.SYSTEM.num_workers, 'pin_memory': cfg.SYSTEM.pin_memory} if use_cuda else {}

    torch.manual_seed(cfg.seed)

    # --------------------------
    # | Prepare datasets
    # --------------------------
    train_set, test_set = dataset.dispatcher(cfg)

    print("Training set contains {} data points.".format(len(train_set)))
    print("Test/Val set contains {} data points.".format(len(test_set)))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.TRAIN.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.TEST.batch_size, shuffle=True, **kwargs)

    # --------------------------
    # | Get ready to learn
    # |  1. Prepare network and loss
    # |  2. Prepare optimizer
    # |  3. Set learning rate
    # --------------------------
    backbone_net = backbone.dispatcher(cfg)
    backbone_net = backbone_net(cfg).to(device)
    feature_shape = backbone_net.get_feature_tensor_shape(device)
    print("Flatten feature length: {}".format(feature_shape))
    post_processor = classifier.dispatcher(cfg, feature_shape)
    
    post_processor = post_processor.to(device)
    
    if cfg.BACKBONE.use_pretrained:
        weight_path = cfg.BACKBONE.pretrained_path
        print("Initializing backbone with pretrained weights from: {}".format(weight_path))
        pretrained_weight_dict = torch.load(weight_path, map_location=device_str)
        backbone_net.load_state_dict(pretrained_weight_dict, strict=False)

    criterion = loss.dispatcher(cfg)

    trainable_params = list(backbone_net.parameters()) + list(post_processor.parameters())

    if cfg.TRAIN.OPTIMIZER.type == "adadelta":
        optimizer = optim.Adadelta(trainable_params, lr = cfg.TRAIN.initial_lr,
                                    weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    elif cfg.TRAIN.OPTIMIZER.type == "SGD":
        optimizer = optim.SGD(trainable_params, lr = cfg.TRAIN.initial_lr, momentum = cfg.TRAIN.OPTIMIZER.momentum,
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay, nesterov = True)
    elif cfg.TRAIN.OPTIMIZER.type == "ADAM":
        optimizer = optim.Adam(trainable_params, lr = cfg.TRAIN.initial_lr, betas = (0.9, 0.999),
                                weight_decay = cfg.TRAIN.OPTIMIZER.weight_decay)
    else:
        raise NotImplementedError("Got unsupported optimizer: {}".format(cfg.TRAIN.OPTIMIZER.type))

    # Prepare LR scheduler
    if cfg.TRAIN.lr_scheduler == "step_down":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = cfg.TRAIN.step_down_on_epoch,
                                                            gamma = cfg.TRAIN.step_down_gamma)
    else:
        raise NotImplementedError("Got unsupported scheduler: {}".format(cfg.TRAIN.lr_scheduler))

    for epoch in range(1, cfg.TRAIN.max_epochs + 1):
        train(cfg, backbone_net, post_processor, criterion, device, train_loader, optimizer, epoch)
        test(cfg, backbone_net, post_processor, criterion, device, test_loader)
        scheduler.step()
        if cfg.save_model:
            torch.save(
                {
                    "backbone": backbone_net.state_dict(),
                    "head": post_processor.state_dict()
                },
                "{0}_epoch{1}.pt".format(cfg.name, epoch)
            )

    if cfg.save_model:
        torch.save(
                {
                    "backbone": backbone_net.state_dict(),
                    "head": post_processor.state_dict()
                },
                "{0}_final.pt".format(cfg.name)
            )


if __name__ == '__main__':
    main()