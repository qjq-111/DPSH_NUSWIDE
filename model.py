import torch.nn as nn


# size = 2                             # LRN的n


'''
class Net(nn.Module):
    def __init__(self, num_binary):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(2),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_binary)
        )

        for m in self.modules():
            if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
                init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''


class Net(nn.Module):
    def __init__(self, original_model, model_name, num_binary):
        super(Net, self).__init__()
        if model_name == 'vgg11':
            self.features = original_model.features
            cl1 = nn.Linear(25088, 4096)
            cl1.weight = original_model.classifier[0].weight
            cl1.bias = original_model.classifier[0].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[3].weight
            cl2.bias = original_model.classifier[3].bias

            self.classifier = nn.Sequential(
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_binary),
            )
            self.model_name = 'vgg11'
        if model_name == 'alexnet':
            self.features = original_model.features
            cl1 = nn.Linear(256 * 6 * 6, 4096)
            cl1.weight = original_model.classifier[1].weight
            cl1.bias = original_model.classifier[1].bias

            cl2 = nn.Linear(4096, 4096)
            cl2.weight = original_model.classifier[4].weight
            cl2.bias = original_model.classifier[4].bias

            self.classifier = nn.Sequential(
                nn.Dropout(),
                cl1,
                nn.ReLU(inplace=True),
                nn.Dropout(),
                cl2,
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_binary),
            )
            self.model_name = 'alexnet'

        '''
        self.model_name = model_name
        if model_name == 'vgg11':
            self.features = original_model.features
            self.classifier = original_model.classifier
            self.classifier[6] = nn.Linear(4096, num_binary)

        if model_name == 'alexnet':
            self.features = original_model.features
            self.classifier = original_model.classifier
            self.classifier[6] = nn.Linear(4096, num_binary)
        '''

    def forward(self, x):
        x = self.features(x)
        if self.model_name == 'vgg11':
            x = x.view(x.size(0), -1)
        if self.model_name == 'alexnet':
            x = x.view(x.size(0), 256*6*6)
        y = self.classifier(x)
        return y
