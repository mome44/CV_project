import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

class CNN_CTC_model(nn.Module):
    def __init__(self, num_char, hidden_size):
        super(CNN_CTC_model, self).__init__()
        self.num_char = num_char
        self.hidden_size = hidden_size
        #self.final_feature_width = final_feature_width
                
        self.features = nn.Sequential(
            #1 Because we use grayscale images
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2, 2),                                     

            nn.Conv2d(128, self.hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )  

        #two linear layers to do the final classification
        self.linear = nn.Linear(self.hidden_size * 12, 256)  # 256×12 = concatenazione H dim
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.hidden_size, self.num_char)

    def forward(self, x):
        #the input size is:  Batch, 1, 48 x 144
        x = self.features(x)   #output size: Batch, 256, 12x36

        #since we have 4 elements, the CTC wants the width first so we have to 
        #put it into the first position
        x = x.permute(3, 0, 1, 2)  # 36, batch, 256, 12 
        #the width so the frames must be more than the number of total characters that
        #we want to encode, so T = width = 36
        x = x.flatten(2)          # 36, batch , 256×12]
        x = self.linear(x)
        x = self.dropout(x)           # 36, batch, 256
        x = self.classifier(x)    # 36, batch, num_char

        return x  #returns a tensor of size [numchar] for each one of the 36 positions
