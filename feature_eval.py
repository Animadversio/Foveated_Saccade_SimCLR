import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

def get_stl10_data_loaders(download, data_dir='./data', batch_size=256):
    train_dataset = datasets.STL10(data_dir, split='train', download=download,
                                    transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=10, drop_last=False, shuffle=True)
    
    test_dataset = datasets.STL10(data_dir, split='test', download=download,
                                    transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                num_workers=10, drop_last=False, shuffle=False)
    return train_loader, test_loader


def get_cifar10_data_loaders(download, shuffle=False, data_dir='./data', batch_size=256):
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=download,
                                                                    transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                        num_workers=10, drop_last=False, shuffle=True)
    
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=download,
                                                                    transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size,
                                                        num_workers=10, drop_last=False, shuffle=False)
    return train_loader, test_loader


def prep_feature_readout(model_path, arch, device="cuda", num_classes=10):
    if type(model_path) is str:
        if arch == 'resnet18':
            model = torchvision.models.resnet18(pretrained=False, num_classes=10).to(device)
        elif arch == 'resnet50':
            model = torchvision.models.resnet50(pretrained=False, num_classes=10).to(device)
        else:
            raise ValueError

        state_dict = torch.load(model_path, map_location=device)['state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('backbone.'):
                if k.startswith('backbone') and not k.startswith('backbone.fc'):
                    # remove prefix
                    state_dict[k[len("backbone."):]] = state_dict[k]
            del state_dict[k]

        log = model.load_state_dict(state_dict, strict=False)
        assert log.missing_keys == ['fc.weight', 'fc.bias']

    elif isinstance(model_path, nn.Module):
        # if input is a
        model = model_path
        feat_dimen = model.fc.in_features
        model.fc = nn.Linear(feat_dimen, num_classes, bias=True, ).to(device)
    else:
        raise ValueError
    
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
        else:
            param.requires_grad = True

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_linear(model, optimizer, train_loader, test_loader, epochs=100, device="cuda"):
    # epochs = 100
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(epochs):
        top1_train_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= (counter + 1)
        top1_accuracy = 0
        top5_accuracy = 0
        for counter, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
        
            top1, top5 = accuracy(logits, y_batch, topk=(1,5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]
        
        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)
        print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")

    return model, optimizer


def feature_evaluation(model_path, arch="resnet18", device="cuda",
                       dataset="STL10", data_dir="./data"):

    model = prep_feature_readout(model_path, arch, device=device)
    if dataset == "STL10":
        train_loader, test_loader = get_stl10_data_loaders(False, data_dir=data_dir, batch_size=256)
    elif dataset == "CIFAR10":
        train_loader, test_loader = get_cifar10_data_loaders(False, shuffle=False,
                                                             data_dir=data_dir, batch_size=256)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0008)
    model, _ = train_linear(model, optimizer, train_loader, test_loader, epochs=100, device="cuda")
    return model


