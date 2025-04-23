import os

def create_log(opt_dict):
    log_train = None
    if not opt_dict['train_config']['find_epoch']:
        log_path = os.path.join(opt_dict['dataset_config']['it_result_path'], opt_dict['model_config']['net_v'], f"b2_{opt_dict['dataset_config']['dataname']}_{opt_dict['train_config']['epochs']}_{opt_dict['dataset_config']['missing_rate']}_{opt_dict['model_config']['model_name']}_{opt_dict['model_config']['net_v']}_{opt_dict['train_config']['lr_max']}_log_train.csv")
        print(log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_train = open(log_path, 'w')

    return log_train


def write_log(log_train, log_out, opt_dict):
    if not opt_dict['train_config']['find_epoch']:
        print(log_out)
        log_train.write(log_out + '\n')
        log_train.flush()
    return log_train
