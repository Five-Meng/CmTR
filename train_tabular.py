import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.utils import AverageMeter, accuracy
from utils.train_tabular_utils import *


def train(model, train_loader, test_loader, device, opt_dict): 
    num_cls = opt_dict['train_config']['num_cls']
    class_longtail_ptpath = get_long_tail_id(opt_dict)
    tail_classes = class_longtail_ptpath[0]  
    long_classes = class_longtail_ptpath[1]  
    log_train = create_log(opt_dict)

    train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
    train_losses = AverageMeter('Train_Loss', ':.4e')
    val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
    val_losses = AverageMeter('Val_loss', ':.4e')
    best_acc = 0

    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    epochs = opt_dict['train_config']['epochs']

    train_all_losses = []
    val_all_losses = []

    for epoch in range(epochs):
        # import ipdb;ipdb.set_trace();
        model.train()
        total_time = time.time()
        train_losses.reset()
        train_acc_v.reset()

        for step, data in enumerate(train_loader):
            tables, labels, masks = data
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)

            batch_size_cur = tables.size(0)
            # import ipdb;ipdb.set_trace();
            logits = model(tables, masks, masks)
            
            loss = loss_function(logits, labels)
            train_losses.update(loss.item(), batch_size_cur)
            train_acc = accuracy(logits, labels)[0]
            train_acc_v.update(train_acc.item(), batch_size_cur)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {data_time:.3f}\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Acc@1 {train_acc_v.val:.4f} ({train_acc_v.avg:.4f})\t'.format(
                epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'], data_time=round((time.time() - total_time), 4), 
                loss=train_losses,
                train_acc_v=train_acc_v))
            log_train = write_log(log_train, log_out, opt_dict)

        train_all_losses.append(train_losses.avg)
        schedule.step()

        model.eval()
        # import ipdb;ipdb.set_trace();

        val_acc_v.reset()
        val_losses.reset()
        val_acc_v, val_losses, total_tail, total_long, correct_tail, correct_long, correct_each_class, total_each_class, tail_acc, long_acc, f1, misclassified = calculate_class_accuracies(
            val_acc_v, val_losses, test_loader, model, loss_function, num_cls, tail_classes, long_classes, opt_dict, device
        )

        log_out2 = f"Tail Accuracy: {tail_acc:.4f} ({correct_tail}/{total_tail})\n" + f"Long Accuracy: {long_acc:.4f} ({correct_long}/{total_long})\n"
        log_train = write_log(log_train, log_out2, opt_dict)
        val_all_losses.append(val_losses.avg)
            
        for cls in range(num_cls):
            correct = correct_each_class[cls]
            total = total_each_class[cls]
            acc_cls = correct / total if total > 0 else 0.0
            log_out3 = f'Class {cls} - Acc@1: {acc_cls:.4f} (Correct: {correct}, Total: {total})'
            log_train = write_log(log_train, log_out3, opt_dict)

        log_out_val = f'Validation - Acc@1: {val_acc_v.avg:.4f}, Loss: {val_losses.avg:.4f}, F1 Score: {f1:.4f}'
        log_train = write_log(log_train, log_out_val, opt_dict)

        best_acc, log_train = save_best_model(log_train, model, misclassified, val_acc_v, best_acc, opt_dict)


    return best_acc





