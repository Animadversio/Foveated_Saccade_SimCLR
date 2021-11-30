import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import yaml


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def get_test_transform(size):
    test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )
    return test_transform

class LogisticRegression(nn.Module):
    """The linear layer to be learnt on top."""
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)


def inference(loader, simclr_model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = simclr_model(x).detach()  # h is the repr, z is the MLP projection.
        
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(simclr_model, train_loader, test_loader, device):
    """ Turn features into a large numpy array, for regression. """
    train_X, train_y = inference(train_loader, simclr_model, device)
    test_X, test_y = inference(test_loader, simclr_model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def eval_train(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    for step, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        # if step % 100 == 0:
        #     print(
        #         f"Step [{step}/{len(loader)}]\t Loss: {loss.item()}\t Accuracy: {acc}"
        #     )

    return loss_epoch, accuracy_epoch


def eval_test(args, loader, simclr_model, model, criterion, optimizer):
    loss_epoch = 0
    accuracy_epoch = 0
    model.eval()
    for step, (x, y) in enumerate(loader):
        model.zero_grad()

        x = x.to(args.device)
        y = y.to(args.device)

        output = model(x)
        loss = criterion(output, y)

        predicted = output.argmax(1)
        acc = (predicted == y).sum().item() / y.size(0)
        accuracy_epoch += acc

        loss_epoch += loss.item()

    return loss_epoch, accuracy_epoch


def get_resnet(arch, device, pretrained=False):
    if arch == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained, num_classes=10).to(device)
    elif arch == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained, num_classes=10).to(device)
    return model


def evaluation(encoder, args, logistic_batch_size=256, logistic_epochs=500, print_every_epoch=50):
    args.image_size = 224
    if "data" in args:  args.dataset_dir = args.data

    proj_head = encoder.fc
    n_features = encoder.fc[0].in_features # input dimensions to the MLP 
    encoder.fc = nn.Identity()

    if args.dataset_name == "stl10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir, split="train", download=True,
            transform=get_test_transform(size=args.image_size),
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir, split="test", download=True,
            transform=get_test_transform(size=args.image_size),
        )
    elif args.dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir, train=True, download=True,
            transform=get_test_transform(size=args.image_size),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir, train=False, download=True,
            transform=get_test_transform(size=args.image_size),
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=logistic_batch_size,
        shuffle=True, drop_last=True, num_workers=args.workers)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=logistic_batch_size,
        shuffle=False, drop_last=True, num_workers=args.workers,)

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        encoder, train_loader, test_loader, args.device
    )
    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, logistic_batch_size
    )
    ## Logistic Regression
    n_classes = 10  # CIFAR-10 / STL-10
    linearhead = LogisticRegression(n_features, n_classes)
    linearhead = linearhead.to(args.device)

    optimizer = torch.optim.Adam(linearhead.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(logistic_epochs):
        loss_epoch, accuracy_epoch = eval_train(
            args, arr_train_loader, encoder, linearhead, criterion, optimizer
        )
        if (1 + epoch) % print_every_epoch == 0:
            print(
    f"Epoch [{epoch}/{logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
            )

    final_train_loss = loss_epoch / len(arr_train_loader)
    final_train_acc  = accuracy_epoch / len(arr_train_loader)
    # final testing
    loss_epoch, accuracy_epoch = eval_test(
        args, arr_test_loader, encoder, linearhead, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )
    final_test_loss = loss_epoch / len(arr_test_loader)
    final_test_acc  = accuracy_epoch / len(arr_test_loader)
    
    encoder.fc = proj_head

    return  final_train_loss, final_train_acc, final_test_loss, final_test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("-dataset-name", default="stl10", type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.ckpt_path = r"E:\Cluster_Backup\SimCLR-runs\Oct01_06-01-34_compute1-exec-209.ris.wustl.edu\checkpoint_0100" \
    #                  r".pth.tar"

    if args.dataset_name == "stl10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir, split="train", download=True,
            transform=get_test_transform(size=args.image_size),
        )
        test_dataset = torchvision.datasets.STL10(
            args.dataset_dir, split="test", download=True,
            transform=get_test_transform(size=args.image_size),
        )
    elif args.dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir, train=True, download=True,
            transform=get_test_transform(size=args.image_size),
        )
        test_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir, train=False, download=True,
            transform=get_test_transform(size=args.image_size),
        )
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.logistic_batch_size,
        shuffle=True, drop_last=True, num_workers=args.workers,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.logistic_batch_size,
        shuffle=False, drop_last=True, num_workers=args.workers,
    )

    encoder = get_resnet(args.resnet, args.device, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer
    state_dict = torch.load(args.ckpt_path, map_location=args.device)['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]
    log = encoder.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    n_features = encoder.fc.in_features
    encoder.fc = nn.Identity()  # get rid of the old fc layer.
    encoder.eval().cuda()

    print("### Creating features from pre-trained context model ###")
    (train_X, train_y, test_X, test_y) = get_features(
        encoder, train_loader, test_loader, args.device
    )

    arr_train_loader, arr_test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logistic_batch_size
    )
    ## Logistic Regression
    n_classes = 10  # CIFAR-10 / STL-10
    model = LogisticRegression(n_features, n_classes)
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.logistic_epochs):
        loss_epoch, accuracy_epoch = eval_train(
            args, arr_train_loader, encoder, model, criterion, optimizer
        )
        if (1 + epoch) % 50 == 0:
            print(
            f"Epoch [{epoch}/{args.logistic_epochs}]\t Loss: {loss_epoch / len(arr_train_loader)}\t Accuracy: {accuracy_epoch / len(arr_train_loader)}"
            )

    # final testing
    loss_epoch, accuracy_epoch = eval_test(
        args, arr_test_loader, encoder, model, criterion, optimizer
    )
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(arr_test_loader)}\t Accuracy: {accuracy_epoch / len(arr_test_loader)}"
    )
