import matplotlib.pyplot as plt
from backbone.model import build_model
from train import train, train_H2T


def train_epoch(model, train_loader, test_loader, device, opt_dict):
    print(opt_dict['train_config']['mode'])
    print(opt_dict['model_config']['net_v'])
    best_accuracies = []

    max_epoch = opt_dict['train_config']['max_epochs']
    for epoch in range(250, max_epoch, 5):
        print(f"现在是Epoch={epoch}")
        opt_dict['train_config']['epochs'] = epoch
        print(f"opt_dict['train_config']['epoch']:{opt_dict['train_config']['epochs']}")
        model = build_model(opt_dict)
        model.to(device)
        if opt_dict['H2T']['h2t']:
            best_acc = train_H2T(model, train_loader, test_loader, device, opt_dict)
        else:    
            best_acc = train(model, train_loader, test_loader, device, opt_dict)

        best_accuracies.append(best_acc)
        print(f"epoch:{epoch}, best_acc:{best_acc}")

    print(f"best_accuracies:{best_accuracies}")
    s = f"Best score:{max(best_accuracies)}, epoch: {best_accuracies.index(max(best_accuracies)) + 10}"
    print(s)

    plt.close()

    print('Finished')


