import matplotlib.pyplot as plt
from train_tabular import train
from backbone_tabular.model_tabular import build_model

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="findfont: Generic family 'sans-serif' not found")


def train_epoch(model, train_loader, test_loader, device, opt_dict):
    print(opt_dict['train_config']['mode'])
    print(opt_dict['model_config']['net_v_tabular'])
    best_accuracies = []

    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(100, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")
        model = build_model(opt_dict)
        model.to(device)
        best_acc = train(model, train_loader, test_loader, device, opt_dict)

        best_accuracies.append(best_acc)
        print(f"epoch:{epoch}, best_acc:{best_acc}")

    print(f"best_accuracies:{best_accuracies}")
    s = f"Best score:{max(best_accuracies)}, epoch: {best_accuracies.index(max(best_accuracies)) + 10}"
    print(s)

    plt.close()

    print('Finished')

