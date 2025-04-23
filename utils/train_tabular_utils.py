import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import accuracy, f1_c


def create_log(opt_dict):
    log_train = None
    log_path = None
    log_path = os.path.join(opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular'], f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['model_config']['model_name']}_{opt_dict['train_config']['lr_max']}_log_train.csv")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_train = open(log_path, 'w')
    print(log_path)
    return log_train


def write_log(log_train, log_out, opt_dict):

    log_train.write(log_out + '\n')
    log_train.flush()
    return log_train




def calculate_class_accuracies(val_acc_v, val_losses, test_loader, model, loss_function, num_cls, tail_classes, long_classes, opt_dict, device):
    correct_each_class = {i: 0 for i in range(num_cls)}
    total_each_class = [0] * num_cls
    correct_long, total_long = 0, 0
    correct_tail, total_tail = 0, 0
    y_true = []
    y_pred = []
    y_score = []  

    misclassified = {i: {j: 0 for j in range(num_cls)} for i in range(num_cls)}

    with torch.no_grad():
        for step, data in enumerate(test_loader):
            tables, labels, masks = data
            tables, labels, masks = tables.to(device), labels.to(device), masks.to(device)
            logits = model(tables, masks, masks)
            val_loss = loss_function(logits, labels)
            val_acc = accuracy(logits, labels)[0]
            val_acc_v.update(val_acc.item(), tables.size(0))
            val_losses.update(val_loss.item(), tables.size(0))
            pred = logits.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(torch.softmax(logits, dim=1).cpu().numpy()) 
            for true_label, pred_label in zip(labels.cpu(), pred.cpu()):
                true_cls = true_label.item()
                pred_cls = pred_label.item()
                total_each_class[true_cls] += 1
                if true_cls == pred_cls:
                    correct_each_class[true_cls] += 1
                
                if true_cls in tail_classes:
                    total_tail += 1
                    if true_cls == pred_cls:
                        correct_tail += 1
                elif true_cls in long_classes:
                    total_long += 1
                    if true_cls == pred_cls:
                        correct_long += 1

                misclassified[true_cls][pred_cls] += 1

    tail_acc = correct_tail / total_tail if total_tail > 0 else 0.0
    long_acc = correct_long / total_long if total_long > 0 else 0.0
    f1 = f1_c(y_true, y_pred, opt_dict['train_config']['num_cls'])

    return val_acc_v, val_losses, total_tail, total_long, correct_tail, correct_long, correct_each_class, total_each_class, tail_acc, long_acc, f1, misclassified




def get_long_tail_id(opt_dict):
    if opt_dict['dataset_config']['dataname'] == 'blood':
        class_longtail_ptpath = torch.load("/data/blood_dvm/data/blood/blood_longtail_target.pt")
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        class_longtail_ptpath = torch.load("/data/blood_dvm/data/dvm/dvm_longtail_500_id.pt")
    return class_longtail_ptpath




def save_best_model(log_train, model, misclassified, val_acc_v, best_acc, opt_dict):
    if val_acc_v.avg > best_acc:
        best_acc = val_acc_v.avg
        # opt_dict['dataset_config']['tabular_result_path'], opt_dict['model_config']['net_v_tabular']
        save_model_path = os.path.join("/data/blood_dvm/data/result/temp/reclsp/", f"{best_acc}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v_tabular']}_{opt_dict['train_config']['lr_max']}_bestmodel.pth")
        log_best_model = f'Saved best model with Acc@1: {best_acc:.4f}\n'
        log_train = write_log(log_train, log_best_model, opt_dict)
        torch.save(model.state_dict(), save_model_path)
    return best_acc, log_train

