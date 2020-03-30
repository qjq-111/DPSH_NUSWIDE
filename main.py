import torch.optim as optim
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

from model import *
from utils import *


# 创建模型
def CreateModel(model_name, num_binary, use_gpu):
    if model_name == 'vgg11':
        original_model = models.vgg11(pretrained=True)
        CNN_net = Net(original_model, model_name, num_binary)
    if model_name == 'alexnet':
        original_model = models.alexnet(pretrained=True)
        CNN_net = Net(original_model, model_name, num_binary)
    if use_gpu:
        CNN_net = CNN_net.cuda()
    return CNN_net


# 设定学习衰减率
def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


class Nus_WideDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        with open(img_path, 'r') as f:
            self.image_path = [path.strip() for path in f]
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_filepath = self.image_path[idx]
        img_filepath = img_filepath.replace('E:/python_project/data', '/data0/qjq/dataset')
        img = Image.open(img_filepath)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[idx]).float()
        # img_filepath = img_filepath.replace('/data0/qjq/dataset/NUS-WIDE/', '')
        return img, label, idx


def dataprocessing(batch_size):
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # image，label所在路径
    test_img_path = 'nus_wide_21/query/image/image_list.txt'
    train_img_path = 'nus_wide_21/train/image/image_list.txt'
    database_img_path = 'nus_wide_21/database/image/image_list.txt'
    test_label_path = 'nus_wide_21/query/label/label.npy'
    train_label_path = 'nus_wide_21/train/label/label.npy'
    database_label_path = 'nus_wide_21/database/label/label.npy'

    database_loader = DataLoader(Nus_WideDataset(database_img_path, database_label_path, transform=transformations),
                                 batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(Nus_WideDataset(train_img_path, train_label_path, transform=transformations),
                              batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(Nus_WideDataset(test_img_path, test_label_path, transform=transformations),
                             batch_size=batch_size, shuffle=False, num_workers=4)
    return test_loader, database_loader, train_loader


def DPSHProcessing(param, num_bit, use_gpu):
    # 获取一些参数
    epochs = param['epochs']
    batch_size = param['batch_size']
    model_name = param['model_name']
    learning_rate = param['learning_rate']
    weight_decay = param['weight_decay']
    lamda = param['lambda']
    MAP = 0

    test_loader, database_loader, train_loader = dataprocessing(batch_size)
    # train_label_path = 'NUS-WIDE/train_label.npy'
    train_label_path = 'nus_wide_21/train/label/label.npy'
    train_label = np.load(train_label_path)
    num_train = train_label.shape[0]
    B = torch.zeros(num_train, num_bit)
    U = torch.zeros(num_train, num_bit)
    train_labels_onehot = torch.from_numpy(train_label).float()
    if use_gpu:
        train_labels_onehot = train_labels_onehot.cuda()

    net = CreateModel(model_name, num_bit, use_gpu)
    optimizer = optim.SGD(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for iter, (img, label, batch_ind) in enumerate(train_loader):
            if use_gpu:
                img, label = img.cuda(), label.cuda()
            S = (label.mm(train_labels_onehot.t()) > 0).float()
            net.zero_grad()
            train_outputs = net(img)
            for i, ind in enumerate(batch_ind):
                # 用这轮产生的batch_size个输出代替之前的那部分
                U[ind, :] = train_outputs.data[i]
                B[ind, :] = torch.sign(train_outputs.data[i])
            Bbatch = torch.sign(train_outputs)
            if use_gpu:
                theta_x = train_outputs.mm(U.cuda().t()) / 2
                logloss = (S.cuda()*theta_x - Logtrick(theta_x, use_gpu)).sum() / (num_train * len(label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(label))
            else:
                theta_x = train_outputs.mm(U.t()) / 2
                logloss = (S*theta_x - Logtrick(theta_x, use_gpu)).sum() / (num_train * len(label))
                regterm = (Bbatch-train_outputs).pow(2).sum() / (num_train * len(label))

            loss = - logloss + lamda * regterm
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (epoch + 1, epochs, epoch_loss / len(train_loader)))
        optimizer = AdjustLearningRate(optimizer, epoch, learning_rate)

    # 测试mAP
        tst_binary, tst_label = compute_result(test_loader, net, use_gpu)
        db_binary, db_label = compute_result(database_loader, net, use_gpu)
        map = compute_mAP(db_binary, db_label, tst_binary, tst_label)
        print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)
        if map > MAP:
            MAP = map
            # save checkpoints
            torch.save(net.state_dict(), 'checkpoints/net1.pth')


def DPSHTest(param, num_bit, use_gpu):
    model_name = param['model_name']
    batch_size = param['batch_size']
    test_loader, database_loader, train_loader = dataprocessing(batch_size)
    net = CreateModel(model_name, num_bit, use_gpu)
    net.load_state_dict(torch.load('checkpoints/net.pth', map_location=lambda storage, location: storage))
    net = Net(48)
    net = net.cuda()

    net.load_state_dict(torch.load('checkpoints/net1.pth'))
    tst_binary, tst_label = compute_result(test_loader, net, use_gpu)
    db_binary, db_label = compute_result(database_loader, net, use_gpu)
    map = compute_mAP(db_binary, db_label, tst_binary, tst_label)
    print('Test MAP: ', map)


if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')
    # bits = [12, 24, 32, 48]
    bits = 48
    param = dict()
    param['epochs'] = 150
    param['batch_size'] = 128
    param['model_name'] = 'alexnet'
    param['learning_rate'] = 0.05
    param['weight_decay'] = 10 ** -5
    param['lambda'] = 50
    use_gpu = torch.cuda.is_available()
    torch.cuda.set_device(1)
    print(str(bits) + 'bits hash code processing ---------------')
    DPSHProcessing(param, bits, use_gpu)
    # DPSHTest(param, bits, use_gpu)

    '''
    bits = 48
    param = dict()
    param['epochs'] = 150
    param['batch_size'] = 128
    param['model_name'] = 'alexnet'
    model_name = param['model_name']
    batch_size = param['batch_size']
    use_gpu = torch.cuda.is_available()
    torch.cuda.set_device(1)
    test_loader, database_loader, train_loader = dataprocessing(batch_size)
    net = CreateModel(model_name, bits, use_gpu)
    # net.load_state_dict(torch.load('checkpoints/net.pth', map_location=lambda storage, location: storage))
    net.load_state_dict(torch.load('checkpoints/net1.pth'))

    bs, img_dir = [], []
    net.eval()
    for img, cls, _, img_path in database_loader:
        img_dir.extend(img_path)
        if use_gpu:
            img = img.cuda()
        bs.append(net(img).data.cpu())
    bs = torch.sign(torch.cat(bs))
    bs = bs.numpy().astype(np.int32)
    np.save('hash_code.npy', bs)
    with open('image_list.txt', mode='w') as f:
        for i_path in img_dir:
            f.write(i_path + '\n')
    '''

