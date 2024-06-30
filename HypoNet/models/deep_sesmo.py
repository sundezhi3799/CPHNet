"""xxx in pytorch

"""
'''xxx in Pytorch.'''

import torch
import torch.nn as nn
import torch.nn.init as init

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [32, 32, 'M', 64, 128, 'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 6
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers+=[nn.Dropout(0.5)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=(3,3), padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

class CNN(nn.Module):

    def __init__(self, channel_number=6,size=1080,num_class=3):

        super().__init__()
        cfg_choose=cfg['B']
        m_number=cfg_choose.count('M')
        self.features=make_layers(cfg_choose,False)
        self.classifier = nn.Sequential(
            nn.Linear(int(cfg_choose[-2]*size/2**m_number*size/2**m_number), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_class)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output



