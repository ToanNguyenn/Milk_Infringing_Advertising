import utils.config as cfg

import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from Classification.utils.utils import *
from model.resnet import ResNet101
from Classification.utils.datasets import MilkDataset

train_accu = []
train_losses = []
eval_losses = []
eval_acc = []


def train(epoch, model, train_loader, optimizer, criterion, device):
    print("\n Epoch : %d" % (epoch + 1))

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0
    for data in tqdm(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        total += labels.size(0)
        running_loss += loss.item()

    accuracy = 100. * correct / total
    train_loss = running_loss / len(train_loader)
    train_accu.append(accuracy)
    train_losses.append(train_loss)
    print('Epochs %d: Train Loss: %.3f | Accuracy: %.3f' % (epoch + 1, train_loss, accuracy))


def test(epoch, model, test_loader, optimizer, criterion, device):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    eval_losses.append(test_loss)
    eval_acc.append(accuracy)

    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accuracy))


def main(model):
    train_ds = MilkDataset('/content/train.csv', f'{cfg.train_dir}', transform=cfg.img_transforms)
    val_ds = MilkDataset('/content/val.csv', f'{cfg.val_dir}', transform=cfg.img_transforms_valid)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True)

    model = model().to(cfg.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if cfg.load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # make_prediction(model, config.val_transforms, r'E:\PycharmProjects\Milk_project\Classification\test', config.DEVICE)
    check_accuracy(val_loader, model, cfg.device)

    for epoch in range(cfg.epochs):
        train(epoch, train_loader=train_loader, model=model, optimizer=optimizer, criterion=loss_fn, device=cfg.device)
        test(epoch, test_loader=val_loader, model=model, optimizer=optimizer, criterion=loss_fn, device=cfg.device)

        # check_accuracy(val_loader, model, cfg.device)
        # checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_model(model, PATH=cfg.save_path)


def pred_testds(model, path):
    test_ds = MilkDataset('/content/test.csv', f'{cfg.test_dir}', transform=cfg.img_transforms_test)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True)
    pred_test_loader(model=model, test_loader=test_loader, path=path)


if __name__ == "__main__":
    # model path
    path = ''
    net = ResNet101(img_channel=3, num_classes=50)
    main(net)
    pred_testds(net, path)