import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

class CNN_CTC_model(nn.Module):
    def __init__(self, num_classes):
        super(CNN_CTC_model, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 48, 144]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # [B, 64, 24, 72]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [B, 128, 24, 72]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # [B, 128, 12, 36]

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# [B, 256, 12, 36]
            nn.ReLU()
        )

        # Riduzione canali → hidden features per carattere
        self.linear = nn.Linear(256 * 12, 256)  # 256×12 = concatenazione H dim
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 1, 48, 144]
        x = self.features(x)  # [B, 256, 12, 36]

        x = x.permute(3, 0, 1, 2)  # [W=36, B, C=256, H=12]
        x = x.flatten(2)          # [T=36, B, 256×12]
        x = self.linear(x)        # [T, B, 256]
        x = self.classifier(x)    # [T, B, num_classes]

        return x  # logits per CTC
