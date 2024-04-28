import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.classifier =nn.Linear(8192, 10)
    
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out