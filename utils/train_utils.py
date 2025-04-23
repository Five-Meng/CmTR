import os
import torch
import torch.nn as nn
from utils.losses import FocalLoss, LDAMLoss
import json
from utils.utils import accuracy, f1_c
import matplotlib.pyplot as plt

def plot_training_and_validation_loss(opt_dict, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid()
    save_path = os.path.join(opt_dict["dataset_config"]["image_result_path"], opt_dict["model_config"]["net_v"], f"{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_TrainValLoss.jpg")

    plt.savefig(save_path)


def create_log(opt_dict):
    log_train = None
    log_path = None
    # print(opt_dict)
    if not opt_dict['H2T']['h2t']:
        if opt_dict['train_config']['lossfunc'] == 'focalloss':
            print("focalloss")
            log_path = os.path.join(opt_dict['dataset_config']['image_result_path'], opt_dict['model_config']['net_v'], f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['train_config']['gamma']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
        else:
            print("*")
            log_path = os.path.join(opt_dict['dataset_config']['image_result_path'], opt_dict['model_config']['net_v'], f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
    else:
        log_path = os.path.join(opt_dict['dataset_config']['image_result_path'], opt_dict['model_config']['net_v'], f"h2t_{opt_dict['H2T']['rho_h2t']}_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_train = open(log_path, 'w')
    print(log_path)
    # return log_train, log_path
    return log_train


def write_log(log_train, log_out, opt_dict):

    log_train.write(log_out + '\n')
    log_train.flush()
    return log_train


def select_loss_function(opt_dict):
    if opt_dict['dataset_config']['dataname'] == 'blood':
        classnum_list = torch.load("/data/blood_dvm/data/blood/blood_ldam_classnum.pt")
    elif opt_dict['dataset_config']['dataname'] == 'dvm':
        classnum_list = torch.load("/data/blood_dvm/data/dvm/dvm_ldam_classnum.pt")
    
    loss_function = {
        'crossentropy': nn.CrossEntropyLoss(),
        'focalloss': FocalLoss(gamma=opt_dict['train_config']['gamma']),
        'ldamloss': LDAMLoss(classnum_list)
    }[opt_dict['train_config']['lossfunc']]

    return loss_function



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
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            val_loss = loss_function(logits, labels)
            val_acc = accuracy(logits, labels)[0]
            val_acc_v.update(val_acc.item(), images.size(0))
            val_losses.update(val_loss.item(), images.size(0))
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

                # misclassified[true_cls][pred_cls] += 1

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
        # if not opt_dict['train_config']['find_epoch'] or opt_dict['train_config']['epochs'] == 10000:
        if not opt_dict['H2T']['h2t']:
            save_model_path = f"{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_best_model.pth"
        else:
            save_model_path = f"h2t_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['train_config']['lossfunc']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_best_model.pth"
        log_best_model = f'Saved best model with Acc@1: {best_acc:.4f}\n'
        log_train = write_log(log_train, log_best_model, opt_dict)
        torch.save(model.state_dict(), os.path.join(opt_dict['dataset_config']['image_result_path'], opt_dict['model_config']['net_v'], save_model_path))
    return best_acc, log_train
    # return log_train


def save_dict_to_json(misclassified, filename):
    with open(filename, 'w') as f:
        json.dump(misclassified, f, indent=4)  