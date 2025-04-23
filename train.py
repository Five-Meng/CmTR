import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import os
from utils.utils import AverageMeter, accuracy
from utils.func_long_tail import H2T
from utils.train_utils import *



# def train(model, train_loader, test_loader, device, opt_dict): 
#     num_cls = opt_dict['train_config']['num_cls']
#     class_longtail_ptpath = get_long_tail_id(opt_dict)
#     tail_classes = class_longtail_ptpath[0]  
#     long_classes = class_longtail_ptpath[1]  
#     log_train = create_log(opt_dict)

#     train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
#     train_losses = AverageMeter('Train_Loss', ':.4e')
#     val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
#     val_losses = AverageMeter('Val_loss', ':.4e')
#     best_acc = 0

#     model.to(device)

#     loss_function = select_loss_function(opt_dict)
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
#     schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
#     epochs = opt_dict['train_config']['epochs']

#     train_all_losses = []
#     val_all_losses = []

#     for epoch in range(epochs):
#         model.train()
#         total_time = time.time()
#         train_losses.reset()
#         train_acc_v.reset()

#         for step, data in enumerate(train_loader):
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)

#             batch_size_cur = images.size(0)
#             logits = model(images)
#             loss = loss_function(logits, labels)
#             train_losses.update(loss.item(), batch_size_cur)
#             train_acc = accuracy(logits, labels)[0]
#             train_acc_v.update(train_acc.item(), batch_size_cur)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
#                        'Time {data_time:.3f}\t'
#                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                        'Acc@1 {train_acc_v.val:.4f} ({train_acc_v.avg:.4f})\t'.format(
#                 epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'], data_time=round((time.time() - total_time), 4), 
#                 loss=train_losses,
#                 train_acc_v=train_acc_v))
#             log_train = write_log(log_train, log_out, opt_dict)

#         train_all_losses.append(train_losses.avg)
#         schedule.step()

#         model.eval()

#         val_acc_v.reset()
#         val_losses.reset()
#         val_acc_v, val_losses, total_tail, total_long, correct_tail, correct_long, correct_each_class, total_each_class, tail_acc, long_acc, f1, misclassified = calculate_class_accuracies(
#             val_acc_v, val_losses, test_loader, model, loss_function, num_cls, tail_classes, long_classes, opt_dict, device
#         )

#         log_out2 = f"Tail Accuracy: {tail_acc:.4f} ({correct_tail}/{total_tail})\n" + f"Long Accuracy: {long_acc:.4f} ({correct_long}/{total_long})\n"
#         log_train = write_log(log_train, log_out2, opt_dict)
#         val_all_losses.append(val_losses.avg)
            
#         for cls in range(num_cls):
#             correct = correct_each_class[cls]
#             total = total_each_class[cls]
#             acc_cls = correct / total if total > 0 else 0.0
#             log_out3 = f'Class {cls} - Acc@1: {acc_cls:.4f} (Correct: {correct}, Total: {total})'
#             log_train = write_log(log_train, log_out3, opt_dict)

#         log_out_val = f'Validation - Acc@1: {val_acc_v.avg:.4f}, Loss: {val_losses.avg:.4f}, F1 Score: {f1:.4f}'
#         log_train = write_log(log_train, log_out_val, opt_dict)

#         best_acc, log_train = save_best_model(log_train, model, misclassified, val_acc_v, best_acc, opt_dict)

#     if not opt_dict['train_config']['find_epoch']:
#         log_train.close()
#         plot_training_and_validation_loss(opt_dict, train_all_losses, val_all_losses)
#         print('Finished')

#     return best_acc


def save_checkpoint(model, optimizer, scheduler, epoch, train_all_losses, val_all_losses, best_acc, log_train_path, opt_dict, filename="checkpoint.pth"):
    """保存 checkpoint，包括模型、优化器、调度器和日志路径"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_all_losses': train_all_losses,
        'val_all_losses': val_all_losses,
        'best_acc': best_acc,
        'log_train_path': log_train_path  # 只存日志文件路径
    }
    torch.save(checkpoint, os.path.join(opt_dict['train_config']['checkpoint_path'], filename))
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, scheduler, opt_dict, device, filename="checkpoint.pth"):
    """加载 checkpoint，恢复训练进度"""
    checkpoint_path = os.path.join(opt_dict['train_config']['checkpoint_path'], filename)
    if os.path.exists(checkpoint_path):
        # checkpoint = torch.load(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device) 
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_all_losses = checkpoint['train_all_losses']
        val_all_losses = checkpoint['val_all_losses']
        best_acc = checkpoint['best_acc']
        log_train_path = checkpoint['log_train_path']  # 读取日志路径
        epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 继续

        log_train = open(log_train_path, "a")

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"Checkpoint loaded, resuming from epoch {epoch}")
        return epoch, train_all_losses, val_all_losses, best_acc, log_train, log_train_path, model, optimizer, scheduler
    else:
        print("No checkpoint found, starting from scratch.")
        log_train, log_train_path = create_log(opt_dict)
        return 0, [], [], 0, log_train, log_train_path, model, optimizer, scheduler


# def train(model, train_loader, test_loader, device, opt_dict):
#     num_cls = opt_dict['train_config']['num_cls']
#     class_longtail_ptpath = get_long_tail_id(opt_dict)
#     tail_classes, long_classes = class_longtail_ptpath[0], class_longtail_ptpath[1]
#     filename = f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_checkpoint_pth"
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
#     schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    
#     #import ipdb;ipdb.set_trace();
#     if 'checkpoint_pth' in opt_dict['train_config']:
#          start_epoch, train_all_losses, val_all_losses, best_acc, log_train, log_train_path, model, optimizer, schedule = load_checkpoint(model, optimizer, schedule, opt_dict, device, filename)
#     else:
#          print("none checkpoint")
#          log_train, log_train_path = create_log(opt_dict)
#          start_epoch, train_all_losses, val_all_losses, best_acc = 0, [], [], 0


#     train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
#     train_losses = AverageMeter('Train_Loss', ':.4e')
#     val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
#     val_losses = AverageMeter('Val_loss', ':.4e')

#     model.to(device)
#     loss_function = select_loss_function(opt_dict)

#     epochs = opt_dict['train_config']['epochs']

#     try:
#         for epoch in range(start_epoch, epochs):  # 从上次中断的 epoch 继续
#             model.train()
#             total_time = time.time()
#             train_losses.reset()
#             train_acc_v.reset()

#             for step, data in enumerate(train_loader):
#                 images, labels = data
#                 images, labels = images.to(device), labels.to(device)

#                 batch_size_cur = images.size(0)
#                 logits = model(images)
#                 loss = loss_function(logits, labels)
#                 train_losses.update(loss.item(), batch_size_cur)
#                 train_acc = accuracy(logits, labels)[0]
#                 train_acc_v.update(train_acc.item(), batch_size_cur)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
#                            'Time {data_time:.3f}\t'
#                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                            'Acc@1 {train_acc_v.val:.4f} ({train_acc_v.avg:.4f})\t'.format(
#                     epoch, step, len(train_loader), lr=optimizer.param_groups[-1]['lr'], 
#                     data_time=round((time.time() - total_time), 4), loss=train_losses, train_acc_v=train_acc_v))
#                 log_train.write(log_out + "\n")

#             train_all_losses.append(train_losses.avg)
#             schedule.step()

#             model.eval()
#             val_acc_v.reset()
#             val_losses.reset()

#             val_acc_v, val_losses, total_tail, total_long, correct_tail, correct_long, correct_each_class, total_each_class, tail_acc, long_acc, f1, misclassified = calculate_class_accuracies(
#                 val_acc_v, val_losses, test_loader, model, loss_function, num_cls, tail_classes, long_classes, opt_dict, device
#             )

#             log_out2 = f"Tail Accuracy: {tail_acc:.4f} ({correct_tail}/{total_tail})\n" + f"Long Accuracy: {long_acc:.4f} ({correct_long}/{total_long})\n"
#             log_train.write(log_out2 + "\n")
#             val_all_losses.append(val_losses.avg)

#             log_out_val = f'Validation - Acc@1: {val_acc_v.avg:.4f}, Loss: {val_losses.avg:.4f}, F1 Score: {f1:.4f}'
#             log_train.write(log_out_val + "\n")

#             best_acc, log_train = save_best_model(log_train, model, misclassified, val_acc_v, best_acc, opt_dict)

#             if 'checkpoint_pth' in opt_dict['train_config']:
#                 print("dont't save checkpoint")
#                 save_checkpoint(model, optimizer, schedule, epoch, train_all_losses, val_all_losses, best_acc, log_train_path, opt_dict, filename=filename)

#     finally:
#         log_train.close() 
#         print('Finished')

#     return best_acc



def train(model, train_loader, test_loader, device, opt_dict): 
    num_cls = opt_dict['train_config']['num_cls']
    class_longtail_ptpath = get_long_tail_id(opt_dict)
    tail_classes = class_longtail_ptpath[0]  
    long_classes = class_longtail_ptpath[1]  
    # import ipdb;ipdb.set_trace();
    log_train = create_log(opt_dict)

    train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
    train_losses = AverageMeter('Train_Loss', ':.4e')
    val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
    val_losses = AverageMeter('Val_loss', ':.4e')
    best_acc = 0

    no_improve_epochs = 0  
    patience = 50  

    model.to(device)

    loss_function = select_loss_function(opt_dict)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    epochs = opt_dict['train_config']['epochs']

    train_all_losses = []
    val_all_losses = []

    for epoch in range(epochs):
        model.train()
        total_time = time.time()
        train_losses.reset()
        train_acc_v.reset()

        for step, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            batch_size_cur = images.size(0)
            logits = model(images)
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
            # import ipdb;ipdb.set_trace();
            log_train = write_log(log_train, log_out, opt_dict)

        train_all_losses.append(train_losses.avg)
        schedule.step()

        model.eval()

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

        if val_acc_v.avg > best_acc:
            no_improve_epochs = 0  
            best_acc, log_train = save_best_model(log_train, model, misclassified, val_acc_v, best_acc, opt_dict)
        else:
            no_improve_epochs += 1 
            
        if no_improve_epochs >= patience:
            print(f'Early stopping after {patience} epochs without improvement')
            log_train = write_log(log_train, f'Early stopping after {patience} epochs without improvement', opt_dict)
            break

    if not opt_dict['train_config']['find_epoch']:
        log_train.close()
        plot_training_and_validation_loss(opt_dict, train_all_losses, val_all_losses)
        print('Finished')

    return best_acc



def train_H2T(model, train_loader, test_loader, device, opt_dict): 
    print(f"================ {opt_dict['H2T']['rho_h2t']} ================")
    num_cls = opt_dict['train_config']['num_cls']
    class_longtail_ptpath = get_long_tail_id(opt_dict)
    tail_classes = class_longtail_ptpath[0]  
    long_classes = class_longtail_ptpath[1]  
    log_train, log_path = create_log(opt_dict)

    train_acc_v = AverageMeter('Train_Acc@1', ':6.2f')
    train_losses = AverageMeter('Train_Loss', ':.4e')
    val_acc_v = AverageMeter('Val_Acc@1', ':6.2f')
    val_losses = AverageMeter('Val_loss', ':.4e')
    best_acc = 0
    patience = 5
    no_improve  = 0

    # save_model_path = "/data/blood_dvm/data/result/end/encoder_image/classification_results/efficientnet-b1/blood_275_crossentropy_efficientnet-b1_1e-3_best_model.pth"
    save_model_path = "/data/blood_dvm/data/result/end/encoder_image/classification_results/efficientnet-b1/dvm_300_crossentropy_efficientnet-b1_3e-3_best_model.pth"
    pre_path = os.path.join(opt_dict['dataset_config']['image_result_path'], opt_dict['model_config']['net_v'], save_model_path)
    print(f"pre_path:{pre_path}")
    checkpoint = torch.load(pre_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    for name, para in model.named_parameters():
        para.requires_grad_(True)

    model.to(device)

    loss_function = select_loss_function(opt_dict)
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=float(opt_dict['train_config']['lr_max']))
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_dict['train_config']['epochs'])
    epochs = opt_dict['train_config']['epochs']

    train_all_losses = []
    val_all_losses = []

    for epoch in range(epochs):
        model.train()
        total_time = time.time()
        train_losses.reset()
        train_acc_v.reset()

        for step, data in enumerate(train_loader):
            images, labels, meta = data
            images, labels = images.to(device), labels.to(device)
            batch_size_cur = images.size(0)
            if opt_dict['H2T']['rho_h2t'] > 0 and 'sample_image' in meta:
                input2 = meta['sample_image'].to(device)
                input2_images = input2
                with torch.no_grad():
                    feat1 = model.forward_split(images, h2t=True)
                    feat2 = model.forward_split(input2_images, h2t=True)
                    feat = H2T(feat1, feat2, rho=opt_dict['H2T']['rho_h2t'])
            logits = model.forward_split(feat, h2t=False)
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
        if(best_acc > val_acc_v.avg):
            no_improve = no_improve + 1
        else:
            no_improve = 0

    if not opt_dict['train_config']['find_epoch']:
        log_train.close()
        plot_training_and_validation_loss(opt_dict, train_all_losses, val_all_losses)
        print('Finished')

    return best_acc
    
